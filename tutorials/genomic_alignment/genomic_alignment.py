import requests
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from mashumaro.mixins.json import DataClassJSONMixin
from flytekit import (
    kwtypes,
    task,
    Resources,
    current_context,
    TaskMetadata,
    dynamic,
    ImageSpec,
    workflow,
)
from flytekit.extras.tasks.shell import OutputLocation, ShellTask, subproc_execute
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory


main_img = ImageSpec(
    name="main",
    platform="linux/amd64",
    python_version="3.11",
    packages=["unionai==0.1.42"],
    conda_channels=["bioconda"],
    conda_packages=[
        "fastp",
        "bowtie2",
    ],
    # builder="fast-builder",
    registry="ghcr.io/unionai",
)


@dataclass
class Reads(DataClassJSONMixin):
    """
    Represents a sequencing reads sample via its associated fastq files.

    This class defines the structure for representing a sequencing sample. It includes
    attributes for the sample name and paths to the read files (R1 and R2).

    Attributes:
        sample (str): The name or identifier of the raw sequencing sample.
        filtered (bool): A boolean value indicating whether the reads have been filtered.
        filt_report (FlyteFile): A FlyteFile object representing the path to the filter report.
        read1 (FlyteFile): A FlyteFile object representing the path to the raw R1 read file.
        read2 (FlyteFile): A FlyteFile object representing the path to the raw R2 read file.
    """

    sample: str
    filtered: bool | None = None
    filt_report: FlyteFile | None = None
    read1: FlyteFile | None = None
    read2: FlyteFile | None = None

    def get_read_fnames(self):
        filt = "filt." if self.filtered else ""
        return (
            f"{self.sample}_1.{filt}fastq.gz",
            f"{self.sample}_2.{filt}fastq.gz",
        )

    def get_report_fname(self):
        return f"{self.sample}_fastq-filter-report.json"

    def from_remote(cls, urls: List[str]):
        samples = {}
        for fp in list(dir.rglob("*fast*")):
            sample = fp.stem.split("_")[0]

            if sample not in samples:
                samples[sample] = Reads(sample=sample)

            if ".fastq.gz" in fp.name or "fasta" in fp.name:
                mate = fp.name.strip(".fastq.gz").strip(".filt").split("_")[-1]
                if "1" in mate:
                    samples[sample].read1 = FlyteFile(path=str(fp))
                elif "2" in mate:
                    samples[sample].read2 = FlyteFile(path=str(fp))
            elif "filter-report" in fp.name:
                samples[sample].filtered = True
                samples[sample].filt_report = FlyteFile(path=str(fp))

        return list(samples.values())


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
        ref = Path("/tmp").joinpath(url.split("/")[-1])
        try:
            response = requests.get(url)
            with open(ref, "wb") as file:
                file.write(response.content)
        except requests.HTTPError as e:
            print(f"HTTP error: {e}")
            raise e
        return cls(ref.name, FlyteDirectory(ref.parent))


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
        sorted (bool): A boolean value indicating whether the SAM file has been sorted.
        deduped (bool): A boolean value indicating whether the SAM file has been deduplicated.
        bqsr_report (FlyteFile): A FlyteFile object representing a report from the Base Quality
            Score Recalibration (BQSR) process.
    """

    sample: str
    aligner: str
    format: str | None = None
    alignment: FlyteFile | None = None
    alignment_idx: FlyteFile | None = None
    alignment_report: FlyteFile | None = None
    sorted: bool | None = None
    deduped: bool | None = None
    bqsr_report: FlyteFile | None = None

    def _get_state_str(self):
        state = f"{self.sample}_{self.aligner}"
        if self.sorted:
            state += "_sorted"
        if self.deduped:
            state += "_deduped"
        return state

    def get_alignment_fname(self):
        return f"{self._get_state_str()}_aligned.{self.format}"


@task(container_image=main_img)
def fetch_assets(ref_url: str, reads: List[str]) -> Tuple[Reference, List[Reads]]:
    """
    Fetch assets from remote URLs.
    """
    ref = Reference.from_remote(url=ref_url)
    samples = Reads.from_remote(urls=reads)
    return ref, samples


@task(
    requests=Resources(mem="2Gi"),
    container_image=main_img,
)
def pyfastp(rs: Reads) -> Reads:
    """
    Perform quality filtering and preprocessing using Fastp on a RawSample.

    This function takes a RawSample object containing raw sequencing data, performs quality
    filtering and preprocessing using the pyfastp tool, and returns a FiltSample object
    representing the filtered and processed data.

    Args:
        rs (RawSample): A RawSample object containing raw sequencing data to be processed.

    Returns:
        FiltSample: A FiltSample object representing the filtered and preprocessed data.
    """
    ldir = Path(current_context().working_directory)
    samp = Reads(rs.sample)
    samp.filtered = True
    o1, o2 = samp.get_read_fnames()
    rep = samp.get_report_fname()
    o1p = ldir.joinpath(o1)
    o2p = ldir.joinpath(o2)
    repp = ldir.joinpath(rep)

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
        "-j",
        repp,
    ]

    subproc_execute(cmd)

    samp.read1 = FlyteFile(path=str(o1p))
    samp.read2 = FlyteFile(path=str(o2p))
    samp.filt_report = FlyteFile(path=str(repp))

    return samp


"""
Generate Bowtie2 index files from a reference genome.

Args:
    ref (FlyteFile): A FlyteFile object representing the input reference file.

Returns:
    FlyteDirectory: A FlyteDirectory object containing the index files.
"""
bowtie2_index = ShellTask(
    name="bowtie2-index",
    debug=True,
    requests=Resources(cpu="4", mem="10Gi"),
    metadata=TaskMetadata(retries=3, cache=True, cache_version=ref_hash),
    container_image=main_img,
    script="""
    mkdir {outputs.idx}
    bowtie2-build {inputs.ref} {outputs.idx}/bt2_idx
    """,
    inputs=kwtypes(ref=FlyteFile),
    output_locs=[
        OutputLocation(var="idx", var_type=FlyteDirectory, location="/tmp/bt2_idx")
    ],
)


@task(
    container_image=main_img,
    requests=Resources(cpu="4", mem="10Gi"),
)
def bowtie2_align_paired_reads(idx: FlyteDirectory, fs: Reads) -> Alignment:
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
    idx.download()
    ldir = Path(current_context().working_directory)

    alignment = Alignment(fs.sample, "bowtie2", "sam")
    al = ldir.joinpath(alignment.get_alignment_fname())
    rep = ldir.joinpath(alignment.get_report_fname())

    cmd = [
        "bowtie2",
        "-x",
        f"{idx.path}/bt2_idx",
        "-1",
        fs.read1,
        "-2",
        fs.read2,
        "-S",
        al,
    ]

    result = subproc_execute(cmd)

    with open(rep, "w") as f:
        f.write(result.error)

    alignment.alignment = FlyteFile(path=str(al))
    alignment.alignment_report = FlyteFile(path=str(rep))
    alignment.sorted = False
    alignment.deduped = False

    return alignment


@dynamic(container_image=main_img)
def bowtie2_align_samples(idx: FlyteDirectory, samples: List[Reads]) -> List[Alignment]:
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
def simple_alignment_wf(seq_dir: FlyteDirectory = "seq_dir_pth"):  # -> List[Alignment]:
    # Prepare raw samples from input directory
    ref, samples = fetch_assets(
        ref_url="https://raw.githubusercontent.com/unionai-oss/unionbio/main/tests/assets/references/GRCh38_short.fasta",
        read_urls=[
            "https://raw.githubusercontent.com/unionai-oss/unionbio/main/tests/sequences/raw/ERR250683-tiny_1.fastq.gz",
            "https://raw.githubusercontent.com/unionai-oss/unionbio/main/tests/sequences/raw/ERR250683-tiny_2.fastq.gz",
        ],
    )

    # # Map out filtering across all samples and generate indices
    # filtered_samples = map_task(pyfastp)(rs=samples)

    # # Generate a bowtie2 index or load it from cache
    # bowtie2_idx = bowtie2_index(ref="ref_loc")

    # # Generate alignments using bowtie2
    # sams = bowtie2_align_samples(idx=bowtie2_idx, samples=filtered_samples)

    # # Return the alignments
    # return sams
