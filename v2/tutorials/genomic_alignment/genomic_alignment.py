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

import subprocess
from pathlib import Path
from dataclasses import dataclass
import flyte
from flyte.io import File, Dir

# ## Defining a Container Image
#
# The previously imported packages are either in the Python Standard Library or included by
# default in the base flyte image used by Union. However, since we want to take advantage of
# a few additional tools, we'll be using flyte.Image to define a custom container image.
# flyte.Image lets us easily specify the base image and add packages or dependencies for our workflow.
# Set the image builder to remote in your flyte config to build the image on union
# image:
#   builder: remote
# 
env = flyte.TaskEnvironment(
    name="alignment-env",
    image=flyte.Image.from_debian_base(
        name="alignment-tutorial",
        python_version=(3, 12)
    )
    .with_apt_packages("fastp", "bowtie2", "ca-certificates")
    .with_uv_project(pyproject_file="pyproject.toml"),
    resources=flyte.Resources(cpu="2", memory="10Gi"),
)

# ## Defining Data Classes
#
# We define three data classes to represent the reference genome, sequencing reads,
# and alignment results.

# Reference genomes are used extensively throughout bioinformatics workflows. We define a
# `Reference` data class to represent a reference genome and its associated index files. The
# class includes attributes for the reference name, the directory containing the reference and
# index files, the index name, and the tool used to create the index. Indices are tool-specific
# files generated from the reference which allow for efficient access.
@dataclass
class Reference:
    """
    Represents a reference FASTA and associated index files.

    Attributes:
        ref_name (str): Name or identifier of the reference file.
        ref_dir (str): reference directory containing the reference and any index files.
        index_name (str): Index string to pass to tools requiring it. Some tools require just the
            ref name and assume index files are in the same dir, others require the index name.
        indexed_with (str): Name of tool used to create the index.
    """

    ref_name: str
    ref_fasta: File
    ref_dir: Dir | None = None
    index_name: str = ""
    indexed_with: str = ""


# Sequencing reads are the raw data generated from a sequencing experiment.
# The digital representation of these sequenced segments are called Reads. 
# The reads are written in the associated FASTQ files.
# The class includes attributes for the sample name and
# paths to the FASTQ files. We assume paired-end reads for this tutorial.
# Though it is possible to modify the code to handle single-end reads as well.
@dataclass
class Reads:
    """
    Represents a sequencing reads sample via its associated FastQ files.

    This class defines the structure for representing a sequencing sample. It includes
    attributes for the sample name and paths to the FASTQ files.

    Attributes:
        sample (str): The name or identifier of the raw sequencing sample.
        read1 (File): A File object representing the path to the raw R1 FASTQ file.
        read2 (File): A File object representing the path to the raw R2 FASTQ file.
    """

    sample: str
    read1: File
    read2: File
       


# Finally, we define an `Alignment` data class to represent an alignment file and its associated
# sample, format, index, and the tool used for the alignment. Alignments are the result of
# mapping the reads back to the reference genome.
@dataclass
class Alignment:
    """
    Represents an alignment file and its associated sample.

    Attributes:
        sample (str): The name or identifier of the sample to which the alignment file belongs.
        aligner (str): The name of the aligner used to generate the alignment file.
        format (str): The format of the alignment file (e.g., SAM, BAM).
        alignment (File): A File object representing the path to the alignment file.
        alignment_idx (File): A File object representing an alignment index file.
    """
    sample: str
    aligner: str
    format: str
    alignment: File | None = None
    alignment_idx: File | None = None


# ## Tasks
#
# We define a series of tasks to perform the following operations:
# 1. Perform quality filtering and preprocessing using FastP
# 2. Generate Bowtie2 index files from a reference genome
# 3. Perform alignment using Bowtie2 on a filtered sample
#
# The first task is quite simple, it simply calls the `from_remote`
# methods on both the `Reference` and `Reads` classes. It will also
# cache these assets, so they won't need to be re-downloaded. This isn't
# as important with the small files we're working with here, but can be
# crucial when working with large reference genomes and sequencing data.


# The first task performs quality filtering and preprocessing using FastP on a Reads object.
# FastP is a performant tool for such operations as removing duplicate, or low-quality reads.
# Since it's a CLI tool, we use python subprocess to call it from within the task.
# Notice how we're also increasing the memory requests for this task so FastP 
# can efficiently process reads from larger FastQ files.
# This is one of Flyte's key strengths: declaring the infrastructure requests alongside
# the task code that depend on them. This allows developers to have clear, versioned, and
# reproducible executions every time.


@env.task
def pyfastp(rs: Reads) -> Reads:
    """
    Perform quality filtering and preprocessing using Fastp on a Reads.

    This function takes a Reads object containing raw sequencing data, performs quality
    filtering and preprocessing using the FastP tool, and returns a Reads object
    representing the filtered and processed data.

    For more information on FastP, visit:
    https://github.com/OpenGene/fastp

    Example FastP command:
    fastp -i in.R1.fq.gz -I in.R2.fq.gz -o out.R1.fq.gz -O out.R2.fq.gz

    Args:
        rs (Reads): A Reads object containing raw sequencing data to be processed.

    Returns:
        Reads: A Reads object representing the filtered and preprocessed data.
    """
    r1_path =  rs.read1.download_sync()
    r2_path =  rs.read2.download_sync()
    r1_filtered = f"{rs.sample}_filtered_1.fastq.gz"
    r2_filtered = f"{rs.sample}_filtered_2.fastq.gz"


    cmd = [
        "fastp",
        "-i",
        r1_path,
        "-I",
        r2_path,
        "-o",
        r1_filtered,
        "-O",
        r2_filtered,
    ]

    subprocess.run(cmd, check=True)

    return Reads(
        sample = rs.sample, 
        read1 = File.from_local_sync(r1_filtered), 
        read2 = File.from_local_sync(r2_filtered)
    )


# Next, we define a task to generate Bowtie2 index files from a reference genome. This task
# takes a Reference object containing the reference genome and adds the index to the same
# FlyteDirectory, while also adding its name and the tool used to generate it. Different tools
# have different conventions around the index name, so it's important to keep track of. As the index
# for a given tool and reference seldom changes, we'll cache this task to avoid regenerating it as well.


@env.task(cache=flyte.Cache(behavior="auto"))
def bowtie2_index(ref: Reference) -> Reference:
    """
    Generate Bowtie2 index files from a reference genome.

    Args:
        ref (Reference): A Reference object representing the reference genome.

    Returns:
        Reference: The same reference object with the index_name and indexed_with attributes set.
    """
    fasta =  ref.ref_fasta.download_sync()
    index_prefix = str(Path(fasta).parent / ref.index_name)
    cmd = [
        "bowtie2-build",
        fasta,
        index_prefix,
    ]
    subprocess.run(cmd, check=True)
    ref.ref_dir =  Dir.from_local_sync(fasta)
    ref.indexed_with = "bowtie2"
    return ref

# The next task performs paired-end alignment using Bowtie 2 on a single sample.
# Similarly to the FastP task, we're wrapping the Bowtie2 CLI in a Python task and using
# subprocess. We then return an `Alignment` object with the path to the alignment file, 
# the sample name, aligner used, and format.


@env.task
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
    idx.ref_dir.download_sync()
    ldir = Path.cwd()
    alignment = Alignment(fs.sample, "bowtie2", "sam")
    al = ldir / f"{fs.sample}_aligned.sam"

    # Construct the Bowtie2 command for paired-end alignment.
    # -x: Specifies the basename of the Bowtie2 index files (created by bowtie2-build). 
    #       This tells Bowtie2 which reference genome to use for alignment.
    # -1: Path to the first set of paired-end reads (FASTQ file, usually R1).
    # -2: Path to the second set of paired-end reads (FASTQ file, usually R2).
    # -S: Output SAM file path. This is where Bowtie2 writes the alignment results.
    cmd = [
        "bowtie2",
        "-x",
        str(Path(idx.ref_fasta.download_sync()).parent / idx.index_name),
        "-1",
        fs.read1.download_sync(),
        "-2",
        fs.read2.download_sync(),
        "-S",
        al,
    ]

    subprocess.run(cmd)

    alignment.alignment = File(path=str(al))

    return alignment


# Finally, we define a dynamic workflow to process samples through the Bowtie2 task above.
# Dynamics are a handy parallelism construct that give your workflow more flexibility via
# the ability to process and arbitrary number of samples. In this case, we're taking a list
# of `Reads` objects and returning a list of `Alignment` objects.


def bowtie2_align_samples(idx: Reference, samples: list[Reads]) -> list[Alignment]:
    """
    Process samples through bowtie2.

    This function takes a Reference object representing a bowtie index and a list of
    Reads objects containing filtered sample data. It performs paired-end alignment
    using bowtie2. It then returns a list of Alignment objects representing the alignment results.

    Args:
        idx (Reference): The Reference object representing the bowtie2 index.
        samples (List[Reads]): A list of Reads objects containing sample data
            to be processed.

    Returns:
        List[Alignment]: A list of Alignment objects representing the alignment results for each sample.
    """
    return [bowtie2_align_paired_reads(idx=idx, fs=sample) for sample in samples]


# ## End-to-End Workflow
#
# We're tying everything together in a final workflow that fetches assets, filters them, generates
# an index, and aligns the samples. This workflow is a simple linear pipeline, but the tasks are
# designed to be modular and reusable. This makes it easy to swap out tools, or add additional
# processing steps as needed. Note that we're also using a `map_task` to parallelize the FastP
# task across all samples. Map tasks are a similar parallelism construct to dynamics, but trade
# some flexibility for improved performance.


@env.task
def alignment_wf() -> list[Alignment]:
    
    ref    = "https://github.com/unionai-oss/unionbio/raw/main/tests/assets/references/GRCh38_short.fasta"
    read_1 = "https://github.com/unionai-oss/unionbio/raw/main/tests/assets/sequences/raw/ERR250683-tiny_1.fastq.gz"
    read_2 = "https://github.com/unionai-oss/unionbio/raw/main/tests/assets/sequences/raw/ERR250683-tiny_2.fastq.gz"

    ref = Reference(
        ref_name="GRCh38_short",
        ref_fasta=File.from_existing_remote(ref),
        index_name = "bt2_idx",
        ref_dir=None,
        indexed_with="",
    )


    samples = [
        Reads(
            sample="ERR250683",
            read1=File.from_existing_remote(read_1),
            read2=File.from_existing_remote(read_2),
        )
    ]

    filtered_samples = [pyfastp(rs=sample) for sample in samples]
    bowtie2_idx = bowtie2_index(ref=ref)
    sams = bowtie2_align_samples(idx=bowtie2_idx, samples=filtered_samples)
    return sams


# You can now run the workflow using the command in the dropdown at the top of the page!
