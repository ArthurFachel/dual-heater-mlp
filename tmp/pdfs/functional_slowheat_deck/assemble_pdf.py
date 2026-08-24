from pathlib import Path
from math import ceil

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[3]
RENDERED = ROOT / "tmp/pdfs/functional_slowheat_deck/rendered"
OUTPUT = ROOT / "output/pdf/functional_slowheat_apresentacao.pdf"
PAGE_W = 960.0
PAGE_H = 540.0
PX_TO_PT = 0.75

URL = {
    "adam": "https://arxiv.org/abs/1412.6980",
    "adamw": "https://arxiv.org/abs/1711.05101",
    "replay": "https://arxiv.org/abs/1902.10486",
    "derpp": "https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html",
    "erace": "https://openreview.net/forum?id=N8MaByOzUfb",
    "agem": "https://arxiv.org/abs/1812.00420",
    "ewc": "https://doi.org/10.1073/pnas.1611835114",
    "si": "https://proceedings.mlr.press/v70/zenke17a.html",
    "lwf": "https://doi.org/10.1007/978-3-319-46493-0_37",
    "distill": "https://arxiv.org/abs/1503.02531",
    "taylor": "https://research.nvidia.com/publication/2017-04_pruning-convolutional-neural-networks-resource-efficient-inference",
    "scenarios": "https://arxiv.org/abs/1904.07734",
    "early": "https://doi.org/10.1016/S0893-6080(98)00010-0",
}


def pdf_rect(x, y, w, h):
    return (
        x * PX_TO_PT,
        (720 - y - h) * PX_TO_PT,
        (x + w) * PX_TO_PT,
        (720 - y) * PX_TO_PT,
    )


def add_link(pdf, url, x, y, w, h):
    pdf.linkURL(url, pdf_rect(x, y, w, h), relative=0, thickness=0)


def catalog_links(urls):
    rows = ceil(len(urls) / 2)
    row_h = 520 / rows
    boxes = []
    for index, url in enumerate(urls):
        col = index // rows
        row = index % rows
        x = 42 + col * (545 + 68)
        y = 145 + row * row_h
        cell_h = row_h - 8
        boxes.append((url, x, y + cell_h - 24, 545, 24))
    return boxes


links = {
    7: [
        (URL["replay"], 42, 487, 505, 30),
        (URL["derpp"], 668, 508, 550, 30),
    ],
    8: [
        (URL["erace"], 42, 440, 535, 30),
        (URL["agem"], 665, 440, 550, 30),
    ],
    9: [
        (URL["ewc"], 42, 451, 350, 28),
        (URL["si"], 441, 451, 350, 28),
        (URL["lwf"], 840, 451, 350, 28),
    ],
    17: catalog_links([
        URL["adamw"], URL["taylor"], URL["taylor"], URL["taylor"],
        URL["taylor"], URL["taylor"], URL["adamw"],
    ]),
    18: catalog_links([
        URL["taylor"], URL["taylor"], URL["adamw"], URL["ewc"],
        URL["taylor"], URL["replay"],
    ]),
    19: catalog_links([
        URL["replay"], URL["distill"], URL["replay"], URL["distill"],
        URL["derpp"], URL["derpp"],
    ]),
    20: catalog_links([
        URL["erace"], URL["agem"], URL["ewc"], URL["si"],
        URL["lwf"], URL["replay"],
    ]),
    21: catalog_links([
        URL["replay"], URL["early"], URL["adamw"], URL["replay"],
        URL["replay"], URL["replay"],
    ]),
}

reference_urls = [
    URL["adam"], URL["adamw"], URL["replay"], URL["derpp"],
    URL["erace"], URL["agem"], URL["ewc"], URL["si"],
    URL["lwf"], URL["distill"], URL["taylor"], URL["scenarios"],
]
links[22] = [
    (url, 42 + (index % 2) * 606, 156 + (index // 2) * 76, 555, 36)
    for index, url in enumerate(reference_urls)
]


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pdf = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
pdf.setTitle("Functional SlowHeat - Metodos e resultados preliminares")
pdf.setAuthor("dual-heater research project")
pdf.setSubject("Aprendizagem continua: SlowHeat, baselines e resultados MNIST")

images = sorted(RENDERED.glob("slide-*.png"))
if len(images) != 23:
    raise RuntimeError(f"Esperadas 23 paginas renderizadas; encontradas {len(images)}")

for page_number, image_path in enumerate(images, start=1):
    pdf.drawImage(ImageReader(str(image_path)), 0, 0, width=PAGE_W, height=PAGE_H)
    for url, x, y, w, h in links.get(page_number, []):
        add_link(pdf, url, x, y, w, h)
    pdf.showPage()

pdf.save()
print(OUTPUT)
