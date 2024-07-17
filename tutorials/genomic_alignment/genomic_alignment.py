# # Genomic Alignment
#
# This tutorial demonstrates how to use Flyte to build a workflow that
# performs genomic alignment on sequencing data. The workflow takes as input
# a reference genome and raw sequencing data, performs quality filtering and
# preprocessing on the raw data, generates an index for the reference genome,
# and aligns the filtered data to the reference genome using the Bowtie 2 aligner.
#
# The tutorial is divided into the following sections:
# 1. Define the container image
# 2. Define the data classes
# 3. Define the tasks
# 4. Define the workflow

import os
import requests
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from mashumaro.mixins.json import DataClassJSONMixin
from flytekit import (
    task,
    Resources,
    current_context,
    dynamic,
    ImageSpec,
    workflow,
    map_task,
)
from flytekit.extras.tasks.shell import subproc_execute
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory

# ## Defining a Container Image
#
# The previously imported packages are either in the Python Standard Library or included by
# default in the base flyte image used by Union. However, since we want to take advantage of
# a few additional tools, we'll be using ImageSpec to define a custom container image.
# ImageSpec lets us easily specify the `bioconda` channel as well as our preprocessing
# packages, `fastp` and our aligner, `bowtie2`. Using ImageSpec here saves us from having
# to manually pull a micromamba binary, set up environments, and install packages.

main_img = ImageSpec(
    name="alignment-tutorial",
    platform="linux/amd64",
    python_version="3.11",
    packages=["unionai==0.1.42"],
    conda_channels=["bioconda"],
    conda_packages=[
        "fastp",
        "bowtie2",
    ],
    builder="fast-builder",
    registry="docker.io/unionbio",
)

# ## Defining Data Classes
#
# We define three data classes to represent the reference genome, sequencing reads, and
# and alignment results. We'll first define a convenience function to download files, which
# we'll use within each dataclass to materialize the appropriate assets from their remote
# locations.


def fetch_file(url: str, local_dir: str) -> Path:
    """
    Downloads a file from the specified URL.

    Args:
        url (str): The URL of the file to download.
        local_dir (Path): The directory where you would like this file saved.

    Returns:
        Path: The local path to the file.

    Raises:
        requests.HTTPError: If an HTTP error occurs while downloading the file.
    """
    url_parts = url.split("/")
    fname = url_parts[-1]
    local_path = Path(local_dir).joinpath(fname)

    try:
        response = requests.get(url)
        with open(local_path, "wb") as file:
            file.write(response.content)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        raise e

    return local_path


# Reference genomes are used extensively throughout bioinformatics workflows. We define a
# `Reference` data class to represent a reference genome and its associated index files. The
# class includes attributes for the reference name, the directory containing the reference and
# index files, the index name, and the tool used to create the index. Indices are tool-specific
# files generated from a the reference which allow for efficient access.


@dataclass
class Reference(DataClassJSONMixin):
    """
    Represents a reference FASTA and associated index files.

    This class captures a directory containing a reference FASTA and optionally it's associated
    index files.

    Attributes:
        ref_name (str): Name or identifier of the raw sequencing sample.
        ref_dir (FlyteDirectory): Directory containing the reference and any index files.
        index_name (str): Index string to pass to tools requiring it. Some tools require just the
        ref name and assume index files are in the same dir, others require the index name.
        indexed_with (str): Name of tool used to create the index.
    """

    ref_name: str
    ref_dir: FlyteDirectory
    index_name: str | None = None
    indexed_with: str | None = None

    @classmethod
    def from_remote(cls, url: str):
        ref_dir = Path("/tmp/reference_genome")
        ref_dir.mkdir(exist_ok=True, parents=True)
        ref = fetch_file(url, ref_dir)
        return cls(ref.name, FlyteDirectory(path=str(ref_dir)))


# Sequencing reads are the raw data generated from a sequencing experiment. In order to
# parallelize processing of the (extremely) long strands of DNA contained in each cell,
# we first split them up into much smaller segments. The digital representation of these
# sequeced segments are called Reads. We define a `Reads` data class to represent a sequencing
# reads sample via its associated fastq files. FastQ files are simply long text files of these
# reads with associated quality scores. The class includes attributes for the sample name and
# paths to the read files (R1 and R2). We define a similar `from_remote` method which fetches
# a list of URLs and constructs a list of `Reads` objects.


@dataclass
class Reads(DataClassJSONMixin):
    """
    Represents a sequencing reads sample via its associated fastq files.

    This class defines the structure for representing a sequencing sample. It includes
    attributes for the sample name and paths to the read files (R1 and R2).

    Attributes:
        sample (str): The name or identifier of the raw sequencing sample.
        read1 (FlyteFile): A FlyteFile object representing the path to the raw R1 read file.
        read2 (FlyteFile): A FlyteFile object representing the path to the raw R2 read file.
    """

    sample: str
    read1: FlyteFile | None = None
    read2: FlyteFile | None = None

    def get_read_fnames(self):
        return (
            f"{self.sample}_1.fastq.gz",
            f"{self.sample}_2.fastq.gz",
        )

    def get_report_fname(self):
        return f"{self.sample}_fastq-filter-report.json"

    @classmethod
    def from_remote(cls, urls: List[str]):
        dl_loc = Path("/tmp/reads")
        dl_loc.mkdir(exist_ok=True, parents=True)

        samples = {}
        for fp in [fetch_file(url, dl_loc) for url in urls]:
            sample = fp.stem.split("_")[0]

            if sample not in samples:
                samples[sample] = Reads(sample=sample)

            if ".fastq.gz" in fp.name or "fasta" in fp.name:
                mate = fp.name.strip(".fastq.gz").strip(".filt").split("_")[-1]
                if "1" in mate:
                    samples[sample].read1 = FlyteFile(path=str(fp))
                elif "2" in mate:
                    samples[sample].read2 = FlyteFile(path=str(fp))

        return list(samples.values())


# Finally, we define an `Alignment` data class to represent an alignment file and its associated
# sample, format, index, and the tool used for the alignment.


@dataclass
class Alignment(DataClassJSONMixin):
    """
    Represents a SAM (Sequence Alignment/Map) file and its associated sample and report.

    This class defines the structure for representing a SAM file along with attributes
    that describe the associated sample and report.

    Attributes:
        sample (str): The name or identifier of the sample to which the SAM file belongs.
        aligner (str): The name of the aligner used to generate the SAM file.
        alignment (FlyteFile): A FlyteFile object representing the path to the alignment file.
        alignment_report (FlyteFile): A FlyteFile object representing an associated report
            for performance of the aligner.
    """

    sample: str
    aligner: str
    format: str | None = None
    alignment: FlyteFile | None = None
    alignment_idx: FlyteFile | None = None

    def get_alignment_fname(self):
        return f"{self.sample}_{self.aligner}_aligned.{self.format}"


# ## Tasks
#
# We define a series of tasks to perform the following operations:
# 1. Fetch assets from remote URLs
# 2. Perform quality filtering and preprocessing using Fastp
# 3. Generate Bowtie2 index files from a reference genome
# 4. Perform paired-end alignment using Bowtie 2 on a filtered sample


@task(container_image=main_img, cache=True, cache_version="4")
def fetch_assets(ref_url: str, read_urls: List[str]) -> Tuple[Reference, List[Reads]]:
    """
    Fetch assets from remote URLs.
    """
    ref = Reference.from_remote(url=ref_url)
    samples = Reads.from_remote(urls=read_urls)
    return ref, samples


@task(
    requests=Resources(mem="2Gi"),
    container_image=main_img,
)
def pyfastp(rs: Reads) -> Reads:
    """
    Perform quality filtering and preprocessing using Fastp on a Reads.

    This function takes a Reads object containing raw sequencing data, performs quality
    filtering and preprocessing using the pyfastp tool, and returns a Reads object
    representing the filtered and processed data.

    Args:
        rs (Reads): A Reads object containing raw sequencing data to be processed.

    Returns:
        Reads: A Reads object representing the filtered and preprocessed data.
    """
    ldir = Path(current_context().working_directory)
    samp = Reads(rs.sample)
    o1, o2 = samp.get_read_fnames()
    o1p = ldir.joinpath(o1)
    o2p = ldir.joinpath(o2)

    cmd = [
        "fastp",
        "-i",
        rs.read1,
        "-I",
        rs.read2,
        "-o",
        o1p,
        "-O",
        o2p,
    ]

    subproc_execute(cmd)

    samp.read1 = FlyteFile(path=str(o1p))
    samp.read2 = FlyteFile(path=str(o2p))

    return samp


@task(
    container_image=main_img,
    requests=Resources(mem="10Gi"),
    cache=True,
    cache_version="4",
)
def bowtie2_index(ref: Reference) -> Reference:
    """
    Generate Bowtie2 index files from a reference genome.

    Args:
        ref (Reference): A Reference object representing the reference genome.

    Returns:
        Reference: The same reference object with the index_name and indexed_with attributes set.
    """
    ref.ref_dir.download()
    idx_name = "bt2_idx"
    cmd = [
        "bowtie2-build",
        f"{ref.ref_dir.path}/{ref.ref_name}",
        f"{ref.ref_dir.path}/{idx_name}",
    ]
    subproc_execute(cmd)
    return Reference(ref.ref_name, FlyteDirectory(ref.ref_dir.path), idx_name, "bowtie2")


@task(
    container_image=main_img,
    requests=Resources(cpu="2", mem="10Gi"),
)
def bowtie2_align_paired_reads(idx: Reference, fs: Reads) -> Alignment:
    """
    Perform paired-end alignment using Bowtie 2 on a filtered sample.

    This function takes a FlyteDirectory object representing the Bowtie 2 index and a
    FiltSample object containing filtered sample data. It performs paired-end alignment
    using Bowtie 2 and returns a Alignment object representing the resulting alignment.

    Args:
        idx (FlyteDirectory): A FlyteDirectory object representing the Bowtie 2 index.
        fs (Reads): A filtered sample Reads object containing filtered sample data to be aligned.

    Returns:
        Alignment: An Alignment object representing the alignment result.
    """
    assert idx.indexed_with == "bowtie2", "Reference index must be generated with bowtie2"

    idx.ref_dir.download()
    ldir = Path(current_context().working_directory)

    alignment = Alignment(fs.sample, "bowtie2", "sam")
    al = ldir.joinpath(alignment.get_alignment_fname())

    print(os.listdir(idx.ref_dir.path))

    cmd = [
        "bowtie2",
        "-x",
        str(Path(idx.ref_dir.path).joinpath(idx.index_name)),
        "-1",
        fs.read1,
        "-2",
        fs.read2,
        "-S",
        al,
    ]

    subproc_execute(cmd)

    alignment.alignment = FlyteFile(path=str(al))

    return alignment


@dynamic(container_image=main_img)
def bowtie2_align_samples(idx: Reference, samples: List[Reads]) -> List[Alignment]:
    """
    Process samples through bowtie2.

    This function takes a FlyteDirectory objects representing a bowtie index and a list of
    Reads objects containing filtered sample data. It performs paired-end alignment
    using bowtie2. It then returns a list of Alignment objects representing the alignment results.

    Args:
        bt2_idx (FlyteDirectory): The FlyteDirectory object representing the bowtie2 index.
        samples (List[Reads]): A list of Reads objects containing sample data
            to be processed.

    Returns:
        List[List[Alignment]]: A list of lists, where each inner list contains alignment
            results (Alignment objects) for a sample, with results from both aligners.
    """
    sams = []
    for sample in samples:
        sam = bowtie2_align_paired_reads(idx=idx, fs=sample)
        sams.append(sam)
    return sams


@workflow
def alignment_wf() -> List[Alignment]:
    # Prepare raw samples from input directory
    ref, samples = fetch_assets(
        ref_url="https://github.com/unionai-oss/unionbio/raw/main/tests/assets/references/GRCh38_short.fasta",
        read_urls=[
            "https://github.com/unionai-oss/unionbio/raw/main/tests/assets/sequences/raw/ERR250683-tiny_1.fastq.gz",
            "https://github.com/unionai-oss/unionbio/raw/main/tests/assets/sequences/raw/ERR250683-tiny_2.fastq.gz",
        ],
    )

    # Map out filtering across all samples and generate indices
    filtered_samples = map_task(pyfastp)(rs=samples)

    # Generate a bowtie2 index or load it from cache
    bowtie2_idx = bowtie2_index(ref=ref)

    # Generate alignments using bowtie2
    sams = bowtie2_align_samples(idx=bowtie2_idx, samples=filtered_samples)

    # Return the alignments
    return sams
