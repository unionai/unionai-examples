# # Genomic Alignment
#
# This tutorial demonstrates how to use Flyte to build a workflow that
# performs genomic alignment on sequencing data. The workflow takes as input
# a reference genome and raw sequencing data, performs quality filtering and
# preprocessing on the raw data, generates an index for the reference genome,
# and aligns the filtered data to the reference genome using the Bowtie 2 aligner.

# {{run-on-union}}

# The tutorial is divided into the following sections:
# 1. Define the container image
# 2. Define the data classes
# 3. Define the tasks
# 4. Define the workflow

# /// script
# requires-python = "3.12"
# dependencies = [
#    "flyte",
#    "requests",
# ]
# main = "alignment_wf"
# params = ""
# ///

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
import flyte
from flyte.io import Dir, File

# ## Defining a Container Image
#
# We define a custom container image using `flyte.Image`. Since we need bioinformatics
# tools — `fastp` for quality filtering and `bowtie2` for alignment — we install them
# via apt. This approach replaces the v1 `ImageSpec` with conda channels.

# {{docs-fragment image}}
main_img = (
    flyte.Image.from_uv_script(
        __file__,
        name="alignment-tutorial",
    )
    .with_apt_packages("fastp", "bowtie2")
)
# {{/docs-fragment image}}

# We define per-task environments with different resource requirements, then a
# top-level `base_env` that declares all of them as dependencies (required because
# `alignment_wf` and `bowtie2_align_samples` call tasks that live in those environments).

# {{docs-fragment envs}}
fetch_env = flyte.TaskEnvironment(
    name="alignment-tutorial-fetch",
    image=main_img,
    cache="auto",
)

fastp_env = flyte.TaskEnvironment(
    name="alignment-tutorial-fastp",
    image=main_img,
    resources=flyte.Resources(memory="2Gi"),
)

index_env = flyte.TaskEnvironment(
    name="alignment-tutorial-index",
    image=main_img,
    resources=flyte.Resources(memory="10Gi"),
    cache="auto",
)

align_env = flyte.TaskEnvironment(
    name="alignment-tutorial-align",
    image=main_img,
    resources=flyte.Resources(cpu=2, memory="10Gi"),
)

base_env = flyte.TaskEnvironment(
    name="alignment-tutorial",
    image=main_img,
    depends_on=[fetch_env, fastp_env, index_env, align_env],
)
# {{/docs-fragment envs}}

# ## Defining Data Classes
#
# We define three data classes to represent the reference genome, sequencing reads,
# and alignment results. We'll first define a convenience function to download files,
# which we'll use within the fetch task to materialize assets from their remote locations.


def fetch_file(url: str, local_dir: str) -> Path:
    """
    Downloads a file from the specified URL.

    Args:
        url (str): The URL of the file to download.
        local_dir (str): The directory where you would like this file saved.

    Returns:
        Path: The local path to the file.

    Raises:
        requests.HTTPError: If an HTTP error occurs while downloading the file.
    """
    url_parts = url.split("/")
    fname = url_parts[-1]
    local_path = Path(local_dir) / fname

    response = requests.get(url)
    with open(local_path, "wb") as file:
        file.write(response.content)

    return local_path


# Reference genomes are used extensively throughout bioinformatics workflows. We define a
# `Reference` data class to represent a reference genome and its associated index files.


# {{docs-fragment dataclasses}}
@dataclass
class Reference:
    """
    Represents a reference FASTA and associated index files.

    Attributes:
        ref_name (str): Name or identifier of the reference file.
        ref_dir (Dir): Directory containing the reference and any index files.
        index_name (str): Index string to pass to tools requiring it.
        indexed_with (str): Name of tool used to create the index.
    """

    ref_name: str
    ref_dir: Dir
    index_name: str | None = None
    indexed_with: str | None = None


# Sequencing reads are the raw data generated from a sequencing experiment.


@dataclass
class Reads:
    """
    Represents a sequencing reads sample via its associated FastQ files.

    Attributes:
        sample (str): The name or identifier of the raw sequencing sample.
        read1 (File): A File object representing the path to the raw R1 read file.
        read2 (File): A File object representing the path to the raw R2 read file.
    """

    sample: str
    read1: File | None = None
    read2: File | None = None

    def get_read_fnames(self):
        return (
            f"{self.sample}_1.fastq.gz",
            f"{self.sample}_2.fastq.gz",
        )


# Finally, we define an `Alignment` data class to represent an alignment file.


@dataclass
class Alignment:
    """
    Represents an alignment file and its associated sample.

    Attributes:
        sample (str): The name or identifier of the sample.
        aligner (str): The name of the aligner used to generate the alignment file.
        format (str): The format of the alignment file (e.g., SAM, BAM).
        alignment (File): A File object representing the path to the alignment file.
    """

    sample: str
    aligner: str
    format: str | None = None
    alignment: File | None = None

    def get_alignment_fname(self):
        return f"{self.sample}_{self.aligner}_aligned.{self.format}"
# {{/docs-fragment dataclasses}}


# ## Tasks
#
# We define a series of tasks to perform the following operations:
# 1. Fetch assets from remote URLs
# 2. Perform quality filtering and preprocessing using FastP
# 3. Generate Bowtie2 index files from a reference genome
# 4. Perform alignment using Bowtie2 on a filtered sample
#
# The first task fetches the reference genome and sequencing reads. It is cached
# so that re-runs skip the download step.


# {{docs-fragment fetch_assets}}
@fetch_env.task
async def fetch_assets(
    ref_url: str, read_urls: List[str]
) -> tuple[Reference, List[Reads]]:
    """
    Fetch assets from remote URLs.
    """
    # Download reference genome
    ref_dir = Path("/tmp/reference_genome")
    ref_dir.mkdir(exist_ok=True, parents=True)
    ref = fetch_file(ref_url, str(ref_dir))
    ref_obj = Reference(
        ref_name=ref.name,
        ref_dir=await Dir.from_local(str(ref_dir)),
    )

    # Download sequencing reads
    dl_loc = Path("/tmp/reads")
    dl_loc.mkdir(exist_ok=True, parents=True)

    samples: dict[str, Reads] = {}
    for url in read_urls:
        fp = fetch_file(url, str(dl_loc))
        sample = fp.stem.split("_")[0]

        if sample not in samples:
            samples[sample] = Reads(sample=sample)

        if ".fastq.gz" in fp.name or "fasta" in fp.name:
            mate = fp.name.strip(".fastq.gz").strip(".filt").split("_")[-1]
            if "1" in mate:
                samples[sample].read1 = await File.from_local(str(fp))
            elif "2" in mate:
                samples[sample].read2 = await File.from_local(str(fp))

    return ref_obj, list(samples.values())
# {{/docs-fragment fetch_assets}}


# The second task performs quality filtering and preprocessing using FastP on a Reads object.
# FastP is a performant tool for removing duplicate or low-quality reads. We increase
# the memory request for this task so FastP can efficiently process reads from larger files.


# {{docs-fragment pyfastp}}
@fastp_env.task
async def pyfastp(rs: Reads) -> Reads:
    """
    Perform quality filtering and preprocessing using Fastp on a Reads object.

    Args:
        rs (Reads): A Reads object containing raw sequencing data to be processed.

    Returns:
        Reads: A Reads object representing the filtered and preprocessed data.
    """
    ldir = Path(tempfile.mkdtemp())
    samp = Reads(rs.sample)
    o1, o2 = samp.get_read_fnames()
    o1p = ldir / o1
    o2p = ldir / o2

    assert rs.read1 is not None and rs.read2 is not None
    r1 = await rs.read1.download()
    r2 = await rs.read2.download()

    cmd = [
        "fastp",
        "-i", str(r1),
        "-I", str(r2),
        "-o", str(o1p),
        "-O", str(o2p),
    ]
    subprocess.run(cmd, check=True)

    samp.read1 = await File.from_local(str(o1p))
    samp.read2 = await File.from_local(str(o2p))

    return samp
# {{/docs-fragment pyfastp}}


# Next, we define a task to generate Bowtie2 index files from a reference genome. As the index
# for a given tool and reference seldom changes, we cache this task.


# {{docs-fragment bowtie2_index}}
@index_env.task
async def bowtie2_index(ref: Reference) -> Reference:
    """
    Generate Bowtie2 index files from a reference genome.

    Args:
        ref (Reference): A Reference object representing the reference genome.

    Returns:
        Reference: The same reference object with the index_name and indexed_with attributes set.
    """
    ref_dir = await ref.ref_dir.download()
    idx_name = "bt2_idx"
    cmd = [
        "bowtie2-build",
        str(Path(str(ref_dir)) / ref.ref_name),
        str(Path(str(ref_dir)) / idx_name),
    ]
    subprocess.run(cmd, check=True)
    return Reference(
        ref.ref_name,
        await Dir.from_local(str(ref_dir)),
        idx_name,
        "bowtie2",
    )
# {{/docs-fragment bowtie2_index}}


# The next task performs paired-end alignment using Bowtie 2 on a single sample.


# {{docs-fragment bowtie2_align}}
@align_env.task
async def bowtie2_align_paired_reads(idx: Reference, fs: Reads) -> Alignment:
    """
    Perform paired-end alignment using Bowtie 2 on a filtered sample.

    Args:
        idx (Reference): A Reference object containing the Bowtie 2 index.
        fs (Reads): A filtered Reads object containing sample data to be aligned.

    Returns:
        Alignment: An Alignment object representing the alignment result.
    """
    assert idx.indexed_with == "bowtie2", "Reference index must be generated with bowtie2"
    assert idx.index_name is not None
    assert fs.read1 is not None and fs.read2 is not None

    ref_dir = await idx.ref_dir.download()
    r1 = await fs.read1.download()
    r2 = await fs.read2.download()

    ldir = Path(tempfile.mkdtemp())
    alignment = Alignment(fs.sample, "bowtie2", "sam")
    al = ldir / alignment.get_alignment_fname()

    cmd = [
        "bowtie2",
        "-x", str(Path(str(ref_dir)) / idx.index_name),
        "-1", str(r1),
        "-2", str(r2),
        "-S", str(al),
    ]
    subprocess.run(cmd, check=True)

    alignment.alignment = await File.from_local(str(al))
    return alignment
# {{/docs-fragment bowtie2_align}}


# In place of the v1 `@dynamic` workflow, we use a plain async task with `asyncio.gather`
# to run alignments for all samples in parallel.


@base_env.task
async def bowtie2_align_samples(
    idx: Reference, samples: List[Reads]
) -> List[Alignment]:
    """
    Process samples through bowtie2 in parallel.

    Args:
        idx (Reference): A Reference object containing the Bowtie 2 index.
        samples (List[Reads]): A list of Reads objects to be aligned.

    Returns:
        List[Alignment]: A list of Alignment objects representing the alignment results.
    """
    tasks = [bowtie2_align_paired_reads(idx=idx, fs=sample) for sample in samples]
    return list(await asyncio.gather(*tasks))


# ## End-to-End Workflow
#
# We tie everything together in a final task that fetches assets, filters them, generates
# an index, and aligns the samples. In place of the v1 `@workflow`, we use a top-level
# `@base_env.task`. Parallelism across samples is achieved with `asyncio.gather`.


# {{docs-fragment workflow}}
@base_env.task
async def alignment_wf() -> List[Alignment]:
    # Prepare raw samples from remote URLs
    ref, samples = await fetch_assets(
        ref_url="https://github.com/unionai-oss/unionbio/raw/main/tests/assets/references/GRCh38_short.fasta",
        read_urls=[
            "https://github.com/unionai-oss/unionbio/raw/main/tests/assets/sequences/raw/ERR250683-tiny_1.fastq.gz",
            "https://github.com/unionai-oss/unionbio/raw/main/tests/assets/sequences/raw/ERR250683-tiny_2.fastq.gz",
        ],
    )

    # Filter all samples in parallel
    filtered_samples = list(
        await asyncio.gather(*[pyfastp(rs=s) for s in samples])
    )

    # Generate a bowtie2 index or load it from cache
    bowtie2_idx = await bowtie2_index(ref=ref)

    # Generate alignments using bowtie2
    sams = await bowtie2_align_samples(idx=bowtie2_idx, samples=filtered_samples)

    return sams
# {{/docs-fragment workflow}}


# You can now run the workflow using the command in the dropdown at the top of the page!

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(alignment_wf)
    print(run.url)
    run.wait()
