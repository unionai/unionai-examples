"""The raw footage: five Blender open movies, all CC BY, and how they are split into chunks.

Tears of Steel and Elephants Dream are read straight from download.blender.org. Blender now
serves Big Buck Bunny and Sintel only as zip archives, which can't be seeked over HTTP, and
Spring isn't on that server, so those three come from the 720p transcodes on Wikimedia
Commons. Every URL is a 720p or smaller rendition that ffmpeg can seek into with HTTP range
requests, so a chunk reads only its own minutes of the film.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    url: str
    duration_s: float
    credit: str
    license: str = "CC BY"


COMMONS = "https://upload.wikimedia.org/wikipedia/commons/transcoded"

# {{docs-fragment sources}}
SOURCES = {
    s.name: s
    for s in [
        Source(
            "big_buck_bunny", "Big Buck Bunny",
            f"{COMMONS}/c/c0/Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.720p.vp9.webm",
            634.5, "(c) copyright 2008, Blender Foundation, peach.blender.org", "CC BY 3.0",
        ),
        Source(
            "sintel", "Sintel",
            f"{COMMONS}/f/f1/Sintel_movie_4K.webm/Sintel_movie_4K.webm.720p.vp9.webm",
            888.0, "(c) copyright Blender Foundation, durian.blender.org", "CC BY 3.0",
        ),
        Source(
            "tears_of_steel", "Tears of Steel",
            "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
            734.0, "(CC) Blender Foundation, mango.blender.org", "CC BY 3.0",
        ),
        Source(
            "elephants_dream", "Elephants Dream",
            "https://download.blender.org/ED/elephantsdream-720-h264-st-aac.mov",
            653.0, "(c) copyright 2006, Blender Foundation, Netherlands Media Art Institute, orange.blender.org", "CC BY 2.5",
        ),
        Source(
            "spring", "Spring",
            f"{COMMONS}/a/a5/Spring_-_Blender_Open_Movie.webm/Spring_-_Blender_Open_Movie.webm.720p.vp9.webm",
            464.0, "(CC) Blender Foundation, studio.blender.org", "CC BY 4.0",
        ),
    ]
}
# {{/docs-fragment sources}}


@dataclass(frozen=True)
class Chunk:
    """A few minutes of one film: the unit of decoding, scene detection, scoring and caching."""

    source: str
    start_s: int
    end_s: int

    @property
    def id(self) -> str:
        return f"{self.source}-{self.start_s:04d}"


# {{docs-fragment chunks}}
def plan_chunks(names: list[str], chunk_s: int, head_s: int = 0, tail_s: int = 0) -> list[Chunk]:
    """Split each film into chunks of ``chunk_s`` seconds.

    With ``head_s`` and ``tail_s`` set, only the film's opening (title cards) and ending (credits)
    are used, which keeps a smoke run short and still gives the text filter work to do.
    """
    chunks = []
    for name in names:
        dur = int(SOURCES[name].duration_s)
        windows = [(0, dur)] if not head_s else [(0, min(head_s, dur - tail_s))] + ([(dur - tail_s, dur)] if tail_s else [])
        for a, b in windows:
            chunks += [Chunk(name, t, min(t + chunk_s, b)) for t in range(a, b, chunk_s) if min(t + chunk_s, b) - t >= 10]
    return chunks
# {{/docs-fragment chunks}}
