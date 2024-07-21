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

REGISTRY = os.getenv("REGISTRY", None)

main_img = ImageSpec(
    name="alignment-tutorial",
    platform="linux/amd64",
    python_version="3.11",
    conda_channels=["bioconda"],
    conda_packages=[
        "fastp",
        "bowtie2",
    ],
    registry=REGISTRY,
)

# ## Defining Data Classes
#
# We define three data classes to represent the reference genome, sequencing reads,
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

    response = requests.get(url)
    with open(local_path, "wb") as file:
        file.write(response.content)

    return local_path


# Reference genomes are used extensively throughout bioinformatics workflows. We define a
# `Reference` data class to represent a reference genome and its associated index files. The
# class includes attributes for the reference name, the directory containing the reference and
# index files, the index name, and the tool used to create the index. Indices are tool-specific
# files generated from the reference which allow for efficient access.


@dataclass
class Reference(DataClassJSONMixin):
    """
    Represents a reference FASTA and associated index files.

    Attributes:
        ref_name (str): Name or identifier of the reference file.
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
# sequenced segments are called Reads. We define a `Reads` data class to represent a sequencing
# reads sample via its associated FastQ files. FastQ files are simply long text files of these
# reads with associated quality scores. The class includes attributes for the sample name and
# paths to the read files (R1 and R2). We define a similar `from_remote` method which fetches
# a list of URLs and constructs a list of `Reads` objects.


@dataclass
class Reads(DataClassJSONMixin):
    """
    Represents a sequencing reads sample via its associated FastQ files.

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
        for url in urls:
            fp = fetch_file(url, dl_loc)
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
# sample, format, index, and the tool used for the alignment. Alignments are the result of
# mapping the reads back to the reference genome. This is necessary to perform downstream analyses
# such as variant calling, gene expression quantification, and more.


@dataclass
class Alignment(DataClassJSONMixin):
    """
    Represents an alignment file and its associated sample.

    Attributes:
        sample (str): The name or identifier of the sample to which the alignment file belongs.
        aligner (str): The name of the aligner used to generate the alignment file.
        format (str): The format of the alignment file (e.g., SAM, BAM).
        alignment (FlyteFile): A FlyteFile object representing the path to the alignment file.
        alignment_idx (FlyteFile): A FlyteFile object representing an alignment index file.
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
# 2. Perform quality filtering and preprocessing using FastP
# 3. Generate Bowtie2 index files from a reference genome
# 4. Perform alignment using Bowtie2 on a filtered sample
#
# The first task is quite simple, it simply calls the `from_remote`
# methods on both the `Reference` and `Reads` classes. It will also
# cache these assets, so they won't need to be re-downloaded. This isn't
# as important with the small files we're working with here, but can be
# crucial when working with large reference genomes and sequencing data.


@task(container_image=main_img, cache=True, cache_version="1.0")
def fetch_assets(ref_url: str, read_urls: List[str]) -> Tuple[Reference, List[Reads]]:
    """
    Fetch assets from remote URLs.
    """
    ref = Reference.from_remote(url=ref_url)
    samples = Reads.from_remote(urls=read_urls)
    return ref, samples


# The second task performs quality filtering and preprocessing using FastP on a Reads object.
# FastP is a performant tool for such operations as removing duplicate, or low-quality reads.
# Since it's a CLI tool, we're wrapping it in a Python task and using the `subproc_execute`
# helper function. This helper is a Flyte-aware wrapper around `subprocess.run` that will
# surface any errors to the Union console. Notice how we're also increasing the memory
# requests for this task so FastP can efficiently process reads from larger FastQ files.
# This is one of Flyte's key strengths: declaring the infrastructure requests alongside
# the task code that depend on them. This allows developers to have clear, versioned, and
# reproducible executions every time.


@task(
    requests=Resources(mem="2Gi"),
    container_image=main_img,
)
def pyfastp(rs: Reads) -> Reads:
    """
    Perform quality filtering and preprocessing using Fastp on a Reads.

    This function takes a Reads object containing raw sequencing data, performs quality
    filtering and preprocessing using the FastP tool, and returns a Reads object
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


# Next, we define a task to generate Bowtie2 index files from a reference genome. This task
# takes a Reference object containing the reference genome and adds the index to the same
# FlyteDirectory, while also adding its name and the tool used to generate it. Different tools
# have different conventions around the index name, so it's important to keep track of. As the index
# for a given tool and reference seldom changes, we'll cache this task to avoid regenerating it as well.


@task(
    container_image=main_img,
    requests=Resources(mem="10Gi"),
    cache=True,
    cache_version="1.0",
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


# The next task performs paired-end alignment using Bowtie 2 on a single sample.
# Similarly to the FastP task, we're wrapping the Bowtie2 CLI in a Python task and using
# the `subproc_execute` helper function. This allows us to unpack the `Reads` and `Reference`
# objects, and download the reference index before running the alignment. We then return an
# `Alignment` object with the path to the alignment file, the sample name, aligner used,
# and format.


@task(
    container_image=main_img,
    requests=Resources(cpu="2", mem="10Gi"),
)
def bowtie2_align_paired_reads(idx: Reference, fs: Reads) -> Alignment:
    """
    Perform paired-end alignment using Bowtie 2 on a filtered sample.

    This function takes a Reference object representing the Bowtie 2 index and a
    Reads object containing filtered sample data. It performs paired-end alignment
    using Bowtie 2 and returns an Alignment object representing the resulting alignment.

    Args:
        idx (Reference): A Reference object containing the Bowtie 2 index.
        fs (Reads): A filtered sample Reads object containing filtered sample data to be aligned.

    Returns:
        Alignment: An Alignment object representing the alignment result.
    """
    assert idx.indexed_with == "bowtie2", "Reference index must be generated with bowtie2"

    idx.ref_dir.download()
    ldir = Path(current_context().working_directory)

    alignment = Alignment(fs.sample, "bowtie2", "sam")
    al = ldir.joinpath(alignment.get_alignment_fname())

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


# Finally, we define a dynamic workflow to process samples through the Bowtie2 task above.
# Dynamics are a handy parallelism construct that give your workflow more flexibility via
# the ability to process and arbitrary number of samples. In this case, we're taking a list
# of `Reads` objects and returning a list of `Alignment` objects.


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


# ## End-to-End Workflow
#
# We're tying everything together in a final workflow that fetches assets, filters them, generates
# an index, and aligns the samples. This workflow is a simple linear pipeline, but the tasks are
# designed to be modular and reusable. This makes it easy to swap out tools, or add additional
# processing steps as needed. Note that we're also using a `map_task` to parallelize the FastP
# task across all samples. Map tasks are a similar parallelism construct to dynamics, but trade
# some flexibility for improved performance.


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


# You can now run the workflow using the command in the dropdown at the top of the page!
