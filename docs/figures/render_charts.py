"""Render the README benchmark charts in one style.

    python docs/figures/render_charts.py          # needs matplotlib

Numbers are copied from the README tables (and the reports they link); a chart and its table must
change together. Each engine has a light color, a standard color, and a dark color:
stock SGLang is orange, TensorRT is grey, vLLM is blue, and AX Engine is green.
The standard green matches the README banner.
"""
from pathlib import Path

from chart_style import bar_chart, percent

OUT = Path(__file__).resolve().parent

SGLANG_LIGHT = "#eea36c"   # stock SGLang AR
SGLANG = "#e36a12"         # stock SGLang speculative route
SGLANG_DARK = "#a34108"    # a second stock SGLang speculative route
VLLM_LIGHT = "#89b1e9"     # stock vLLM AR
VLLM = "#3b7ddb"           # stock vLLM speculative route
VLLM_DARK = "#1a4e9a"      # a second stock vLLM speculative route
EDGE_LIGHT = "#c5cad3"     # TensorRT AR
EDGE = "#8b939e"           # TensorRT speculative route
EDGE_DARK = "#3e4650"      # TensorRT tree
# A fourth TensorRT route sits between the standard and dark greys.
EDGE_MID = "#646c76"
AX_LIGHT = "#91d7b7"
AX = "#37b67c"             # standard green, the README banner mark
AX_DARK = "#0b5d36"
# A fourth AX route sits between the standard and dark greens.
AX_MID = "#1f8a56"
WORKLOADS = ["Code", "Math", "Chat"]


def split(*cells):
    """'a / b / c' strings to per-workload floats; None for a lane that has no result yet."""
    return [[None] * 3 if cell is None else [float(x.replace(",", "")) for x in cell.split(" / ")] for cell in cells]


def main():
    bar_chart(
        OUT / "b300-flash-next-bf16-throughput.svg",
        title="Two B300 GPUs · Qwen3.8-Flash-Next BF16",
        subtitle="SGLang 0.5.23 · TP2 · second pass · 24 greedy prompts per workload · 256-token limit",
        series=[("Stock SGLang AR", SGLANG_LIGHT), ("Stock SGLang MTP3", SGLANG), ("AX Engine MTP7", AX_DARK)],
        panels=[
            {"title": "Concurrency 1", "categories": WORKLOADS,
             "values": [[184.9, 206.8, 207.2], [306.6, 353.9, 294.4], [390.0, 409.9, 268.0]]},
            {"title": "Concurrency 4", "categories": WORKLOADS,
             "values": [[389.6, 550.4, 534.1], [520.7, 649.4, 584.5], [615.4, 735.2, 521.1]]},
            {"title": "Concurrency 16", "categories": WORKLOADS,
             "values": [[794.5, 1258.0, 1069.6], [1199.7, 1606.1, 1272.6], [1649.8, 1708.0, 1169.6]]},
        ],
        xlabel="End-to-end output tokens per second · same scale in all panels",
        description="Stock AR, stock MTP3 and AX Engine MTP7 at concurrency 1, 4 and 16 for code, math and chat. "
                    "MTP7 leads on code and math; stock MTP3 leads on chat.",
    )

    bar_chart(
        OUT / "b200x2-flash-next-bf16-throughput.svg",
        title="Two B200 GPUs · Qwen3.8-Flash-Next BF16",
        subtitle="TP2 · PLE in pinned host memory · 24 greedy prompts per workload · 256-token limit · two passes",
        series=[("Stock SGLang AR", SGLANG_LIGHT), ("Stock vLLM AR", VLLM_LIGHT), ("Stock SGLang MTP3", SGLANG),
                ("Stock vLLM MTP3", VLLM), ("AX Engine MTP7", AX_DARK)],
        panels=[
            {"title": "Concurrency 1", "categories": WORKLOADS,
             "values": split("215.9 / 230.1 / 227.9", "220.1 / 223.6 / 223.3", "406.7 / 430.3 / 336.9",
                             "508.9 / 483.5 / 369.5", "535.9 / 506.6 / 303.8")},
            {"title": "Concurrency 4", "categories": WORKLOADS,
             "values": split("549.2 / 644.8 / 615.0", "598.6 / 622.4 / 597.8", "877.1 / 979.5 / 754.3",
                             "1,260.3 / 1,270.7 / 945.0", "943.2 / 1,029.8 / 654.3")},
            {"title": "Concurrency 16", "categories": WORKLOADS,
             "values": split("980.5 / 1,301.5 / 1,239.6", "1,233.5 / 1,383.4 / 1,278.4", "1,042.3 / 2,117.8 / 1,599.7",
                             "2,589.4 / 2,671.9 / 1,943.0", "2,078.6 / 2,230.6 / 1,443.4")},
        ],
        xlabel="End-to-end output tokens per second · same scale in all panels",
        description="Stock SGLang AR, stock vLLM AR, stock SGLang MTP3, stock vLLM MTP3 and AX Engine MTP7 at "
                    "concurrency 1, 4 and 16. Stock vLLM MTP3 leads at C4 and C16; AX MTP7 leads code and math at C1.",
    )

    bar_chart(
        OUT / "b200x2-flash-next-bf16-prefill-graph.svg",
        title="Two B200 GPUs · Qwen3.8-Flash-Next BF16 · one pod",
        subtitle="TP2 · PLE in pinned host memory · 24 greedy prompts per workload · 256-token limit · "
                 "passes pooled",
        series=[("Stock SGLang MTP3, eager prefill", SGLANG_LIGHT), ("Stock SGLang MTP3", SGLANG),
                ("Stock vLLM MTP3", VLLM), ("AX Engine MTP7", AX_DARK)],
        panels=[
            {"title": "Concurrency 1", "categories": WORKLOADS,
             "values": split("309.0 / 353.0 / 285.5", "418.2 / 409.3 / 319.2", "364.7 / 440.1 / 314.6",
                             "549.4 / 505.7 / 311.1")},
            {"title": "Concurrency 4", "categories": WORKLOADS,
             "values": split("544.3 / 705.5 / 572.6", "969.9 / 948.2 / 781.1", "884.3 / 1,126.1 / 777.3",
                             "1,073.5 / 1,090.0 / 679.0")},
            {"title": "Concurrency 16", "categories": WORKLOADS,
             "values": split("752.0 / 1,567.3 / 1,125.0", "1,932.9 / 2,127.4 / 1,565.9",
                             "2,096.2 / 2,432.1 / 1,679.6", "1,978.8 / 2,490.2 / 1,442.4")},
        ],
        xlabel="End-to-end output tokens per second · same scale in all panels",
        note="Eager prefill: the earlier launch with --disable-prefill-cuda-graph. The other SGLang lanes use "
             "SGLang's default prefill CUDA graphs.",
        description="On one pod, prefill CUDA graphs raise stock SGLang MTP3 and AX Engine MTP7 to stock vLLM "
                    "MTP3's level at four and sixteen clients; AX MTP7 leads code at one and four clients.",
    )

    bar_chart(
        OUT / "rtx-5090-27b-nvfp4-prefill-graph.svg",
        title="RTX 5090 · Qwen3.8-27B NVFP4 · one pod",
        subtitle="128 greedy output tokens per request · 8 prompts at one request, 24 at two per workload · "
                 "two servers per lane · cute-dsl FP4 GEMMs",
        series=[("Stock SGLang DFlash2", SGLANG), ("Stock vLLM DFlash2", VLLM), ("AX tree 32", AX),
                ("AX GDN replay", AX_DARK)],
        panels=[
            {"title": "1 concurrent request", "categories": WORKLOADS,
             "values": split("294.9 / 237.2 / 155.1", "245.3 / 213.6 / 173.1", "389.4 / 318.0 / 186.4",
                             "317.0 / 259.4 / 160.3")},
            {"title": "2 concurrent requests", "categories": WORKLOADS,
             "values": split("378.0 / 393.3 / 234.9", "316.2 / 345.9 / 231.2", "418.2 / 421.8 / 243.7",
                             "402.9 / 405.5 / 255.7")},
        ],
        xlabel="End-to-end output tokens per second · same scale in both panels",
        note="SGLang lanes use prefill CUDA graphs. First sweep of every server dropped (both engines).",
        description="On one RTX 5090 the AX tree leads stock vLLM DFlash2 on every workload at one and two "
                    "requests (+59/+49/+8% and +32/+22/+5%).",
    )

    bar_chart(
        OUT / "rtx-pro-6000-27b-nvfp4-prefill-graph.svg",
        title="RTX PRO 6000 · Qwen3.8-27B NVFP4 · one pod",
        subtitle="128 greedy output tokens per request · 8 prompts at C1, 24 at C8 per workload · "
                 "two servers per lane · cute-dsl FP4 GEMMs",
        series=[("Stock SGLang DFlash2", SGLANG), ("Stock vLLM DFlash2", VLLM), ("AX tree 32", AX),
                ("AX GDN replay", AX_DARK)],
        panels=[
            {"title": "1 concurrent client", "categories": WORKLOADS,
             "values": split("318.4 / 258.6 / 168.8", "294.1 / 283.4 / 179.5", "431.4 / 363.4 / 209.5",
                             "329.4 / 267.1 / 174.9")},
            {"title": "8 concurrent clients", "categories": WORKLOADS,
             "values": split("1,236.8 / 1,127.7 / 678.8", "897.9 / 926.7 / 690.1", "1,151.8 / 1,216.0 / 684.9",
                             "1,194.8 / 1,313.8 / 790.1")},
        ],
        xlabel="End-to-end output tokens per second · same scale in both panels",
        note="AX lanes admit one DFlash prefill at a time (launch default at 8 slots). First sweep of every "
             "server dropped (both engines).",
        description="On one RTX PRO 6000 the AX tree leads stock vLLM DFlash2 on every workload at one client "
                    "and AX GDN replay on every workload at eight.",
    )

    bar_chart(
        OUT / "b200-27b-nvfp4-prefill-graph.svg",
        title="B200 · Qwen3.8-27B NVFP4 · one pod",
        subtitle="128 greedy output tokens per request · 8 prompts at C1, 24 at C8 per workload · "
                 "two servers per lane · cute-dsl FP4 GEMMs",
        series=[("Stock SGLang DFlash2", SGLANG), ("Stock vLLM DFlash2", VLLM), ("AX tree 32", AX),
                ("AX GDN replay", AX_DARK)],
        panels=[
            {"title": "1 concurrent client", "categories": WORKLOADS,
             "values": split("692.7 / 581.7 / 370.2", "838.4 / 695.6 / 451.6", "945.6 / 766.5 / 459.1",
                             "717.1 / 645.7 / 406.2")},
            {"title": "8 concurrent clients", "categories": WORKLOADS,
             "values": split("2,271.3 / 1,976.9 / 1,329.0", "2,660.1 / 2,851.6 / 1,819.6",
                             "2,634.3 / 2,551.6 / 1,538.8", "2,613.2 / 2,656.8 / 1,707.0")},
        ],
        xlabel="End-to-end output tokens per second",
        note="AX lanes: launch defaults (one DFlash prefill admitted at a time, FlashInfer GDN prefill). "
             "First sweep of every server dropped (both engines).",
        share_scale=False,
        description="On one B200 the AX tree leads stock vLLM DFlash2 on every workload at one client; at eight "
                    "clients AX GDN replay is within 2-7% of vLLM.",
    )

    bar_chart(
        OUT / "b200-27b-nvfp4-serving-retest.svg",
        title="B200 · Qwen3.8-27B NVFP4",
        subtitle="128 greedy output tokens per request · two passes per route · 8 prompts at C1, 24 at C8 per workload",
        series=[("Stock AR", SGLANG_LIGHT), ("Stock DFlash2", SGLANG), ("AX tree/chain", AX), ("AX GDN replay", AX_DARK)],
        panels=[
            {"title": "1 concurrent client", "categories": WORKLOADS,
             "values": split("136.0 / 136.0 / 136.4", "448.5 / 408.1 / 281.3", "632.5 / 516.5 / 333.8",
                             "456.6 / 415.3 / 286.6")},
            {"title": "8 concurrent clients", "categories": WORKLOADS,
             "values": split("870.4 / 851.6 / 833.6", "1,438.4 / 1,172.9 / 903.2", "1,503.0 / 1,206.6 / 790.1",
                             "1,599.5 / 1,428.3 / 949.6")},
        ],
        xlabel="End-to-end output tokens per second",
        note="A separate warmed AR check reached 953.0 tok/s on chat at 8 clients.",
        share_scale=False,
        description="End-to-end throughput at one and eight concurrent clients by workload and serving route.",
    )

    bar_chart(
        OUT / "b200-27b-nvfp4-throughput.svg",
        title="B200 · Qwen3.8-27B NVFP4 · earlier 1,024-token study",
        subtitle="64 prompts per workload · 1,024-token limit · 64 server slots · mean of two passes",
        series=[("Stock SGLang + DFlash2 block 8", SGLANG), ("SGLang + ax-engine-sglang", AX_DARK)],
        panels=[
            {"title": "Concurrency 1 · AX tree 16", "categories": WORKLOADS,
             "values": [[816, 733, 401], [1160, 971, 458]]},
            {"title": "Concurrency 16 · AX chain 8", "categories": WORKLOADS,
             "values": [[3628, 4670, 2980], [4379.5, 4989, 3110]]},
        ],
        xlabel="Output tokens per second · the panels use different scales",
        share_scale=False, fmt=lambda v: f"{v:,.1f}".removesuffix(".0"),
        description="Stock DFlash2 and the add-on on code, math and chat at concurrency one and sixteen.",
    )

    bar_chart(
        OUT / "b200-dflash2-recurrent-replay-throughput.svg",
        title="B200 · Qwen3.8-27B NVFP4 chat",
        subtitle="DFlash2 width 8 · concurrency 16 · 32 slots · 64 chat prompts",
        series=[("Stock GDN snapshots", SGLANG), ("AX recurrent GDN replay (benchmark overlay)", AX_DARK)],
        panels=[{"categories": ["Chat · C16"], "values": [[2685.6], [2957.3]]}],
        xlabel="Output tokens per second", legend_columns=2,
        description="Stock DFlash2 reached 2,685.6 output tokens per second and the recurrent GDN replay overlay "
                    "2,957.3.",
    )

    bar_chart(
        OUT / "b200-flash-next-nvfp4-serving-retest.svg",
        title="B200 · Qwen3.8-Flash-Next NVFP4",
        subtitle="128 greedy output tokens per request · two passes per route · 8 prompts at C1, 24 at C8 per workload",
        series=[("Stock AR", SGLANG_LIGHT), ("Stock MTP3", SGLANG), ("AX MTP3", AX), ("AX MTP7", AX_DARK)],
        panels=[
            {"title": "1 concurrent client", "categories": WORKLOADS,
             "values": split("136.9 / 136.4 / 130.6", "304.9 / 273.3 / 223.4", "302.8 / 270.9 / 213.3",
                             "389.7 / 297.1 / 208.9")},
            {"title": "8 concurrent clients", "categories": WORKLOADS,
             "values": split("614.7 / 717.4 / 594.2", "720.7 / 947.0 / 684.5", "745.1 / 976.8 / 715.4",
                             "1,126.9 / 978.8 / 607.5")},
        ],
        xlabel="End-to-end output tokens per second", note="All routes used pinned-host FP8 PLE.",
        share_scale=False,
        description="End-to-end throughput at one and eight concurrent clients by workload and serving route.",
    )

    bar_chart(
        OUT / "b200-flash-next-mtp-throughput.svg",
        title="B200 · Qwen3.8-Flash-Next NVFP4 · earlier MTP study",
        subtitle="Concurrency 1 · second pass · 256-token limit · SGLang 0.5.23",
        series=[("Stock SGLang AR", SGLANG_LIGHT), ("Stock SGLang MTP3", SGLANG), ("MTP7 + AX Engine QSA fix", AX_DARK)],
        panels=[{"categories": WORKLOADS,
                 "values": split("204.9 / 212.5 / 211.7", "435.2 / 432.3 / 334.4", "547.9 / 487.7 / 296.8")}],
        xlabel="Output tokens per second",
        description="MTP7 with the QSA fix leads on code and math. Stock MTP3 leads on chat.",
    )

    retest = {"subtitle": "128 greedy output tokens per request · two passes per route",
              "xlabel": "End-to-end output tokens per second", "share_scale": False}
    bar_chart(
        OUT / "rtx-pro-6000-27b-bf16-retest.svg",
        title="RTX PRO 6000 · Qwen3.8-27B BF16",
        series=[("Stock AR", SGLANG_LIGHT), ("Stock DFlash2", SGLANG), ("TensorRT-LLM DFlash2", EDGE),
                ("Stock vLLM DFlash2", VLLM), ("AX tree / chain", AX_DARK)],
        panels=[
            {"title": "1 concurrent client", "categories": WORKLOADS,
             "values": split("27.1 / 27.2 / 27.2", "132.0 / 120.9 / 70.5", "133.1 / 119.3 / 73.8",
                             "135.3 / 120.6 / 73.7", "184.3 / 163.2 / 94.1")},
            {"title": "8 concurrent clients", "categories": WORKLOADS,
             "values": split("199.4 / 200.5 / 202.4", "487.5 / 567.6 / 333.7", "529.5 / 720.6 / 402.4",
                             "520.8 / 664.0 / 382.0", "510.8 / 572.0 / 335.1")},
        ],
        description="BF16 throughput at one and eight clients for stock AR, stock DFlash2, TensorRT-LLM DFlash2, "
                    "stock vLLM DFlash2 and the AX tree/chain.",
        **retest,
    )
    nvfp4 = [("1 concurrent client", "54.2 / 54.1 / 54.4", "229.3 / 188.5 / 125.2", "246.0 / 210.3 / 128.9",
              "279.8 / 266.4 / 154.5", "324.7 / 270.9 / 152.2", "235.8 / 195.2 / 128.9"),
             ("8 concurrent clients", "376.0 / 377.3 / 367.0", "819.2 / 760.5 / 505.3", "878.7 / 1,178.0 / 605.3",
              "1,002.3 / 980.1 / 704.0", "907.7 / 823.3 / 487.0", "996.8 / 898.1 / 580.8")]
    bar_chart(
        OUT / "rtx-pro-6000-27b-nvfp4-retest.svg",
        title="RTX PRO 6000 · Qwen3.8-27B NVFP4",
        series=[("Stock AR", SGLANG_LIGHT), ("Stock DFlash2", SGLANG), ("TensorRT-LLM DFlash2", EDGE),
                ("Stock vLLM DFlash2", VLLM), ("AX tree / chain", AX), ("AX GDN replay", AX_DARK)],
        panels=[{"title": t, "categories": WORKLOADS, "values": split(*cells)} for t, *cells in nvfp4],
        description="NVFP4 throughput at one and eight clients for stock AR, stock DFlash2, TensorRT-LLM DFlash2, "
                    "stock vLLM DFlash2, the AX tree/chain and AX GDN replay.",
        **retest,
    )
    bf16_c8 = split("199.4 / 200.5 / 202.4", "487.5 / 567.6 / 333.7", "529.5 / 720.6 / 402.4",
                    "520.8 / 664.0 / 382.0", "510.8 / 572.0 / 335.1")
    bar_chart(
        OUT / "rtx-pro-6000-27b-retest-throughput.svg",
        title="RTX PRO 6000 · Qwen3.8-27B at concurrency 8",
        subtitle="128 greedy output tokens per request · two passes per route",
        series=[("Stock SGLang AR", SGLANG_LIGHT), ("Stock DFlash2, width 8", SGLANG),
                ("TensorRT-LLM DFlash2, 7 draft tokens", EDGE), ("Stock vLLM DFlash2, width 8", VLLM),
                ("AX tree 32 / chain 8", AX), ("AX GDN replay, width 8", AX_DARK)],
        panels=[{"title": "BF16", "categories": WORKLOADS, "values": bf16_c8 + [[None, None, None]],
                 "missing": "not run"},
                {"title": "NVFP4", "categories": WORKLOADS, "values": split(*nvfp4[1][1:])}],
        xlabel="End-to-end output tokens per second",
        description="End-to-end output token throughput with eight concurrent clients.",
    )

    # H200 (SM90): BF16, FP8 and NVFP4 through Marlin W4A16; TensorRT-LLM NVFP4 needs Blackwell (SM100/SM120).
    h200 = [("Stock SGLang AR", SGLANG_LIGHT,
             ("67.4 / 67.3 / 67.3", "93.8 / 93.9 / 94.0", "102.4 / 102.4 / 102.4"),
             ("457.4 / 463.7 / 451.0", "611.8 / 612.2 / 637.4", "642.5 / 643.9 / 672.3")),
            ("Stock SGLang DFlash2", SGLANG,
             ("268.9 / 244.4 / 153.0", "311.3 / 289.6 / 189.1", "342.7 / 304.5 / 204.2"),
             ("1,029.1 / 946.3 / 619.4", "1,122.1 / 895.0 / 690.7", "1,063.8 / 873.5 / 609.4")),
            ("TensorRT-LLM DFlash2", EDGE,
             ("286.2 / 259.8 / 164.7", "350.1 / 316.4 / 198.9", None),
             ("931.7 / 1,011.1 / 702.6", "1,083.0 / 1,115.2 / 721.7", None)),
            ("Stock vLLM DFlash2", VLLM,
             ("309.1 / 285.5 / 175.9", "402.2 / 373.7 / 236.9", "465.7 / 429.9 / 282.2"),
             ("1,084.5 / 1,313.6 / 830.1", "1,160.3 / 1,359.2 / 1,018.3", "1,357.3 / 1,382.4 / 971.8")),
            ("AX tree 32 / chain 8", AX_DARK,
             ("366.2 / 325.4 / 193.3", "410.5 / 352.4 / 226.1", "409.5 / 336.3 / 221.3"),
             ("1,061.0 / 955.4 / 626.7", "1,132.4 / 938.0 / 705.0", "1,071.7 / 913.2 / 619.9"))]
    for slug, clients, col in (("c1", "1 concurrent client", 2), ("c8", "8 concurrent clients", 3)):
        bar_chart(
            OUT / f"h200-27b-{slug}-throughput.svg",
            title=f"H200 · Qwen3.8-27B · {clients}",
            subtitle="128 greedy output tokens per request · two passes per route · "
                     + ("8 prompts" if col == 2 else "24 prompts") + " per workload",
            series=[(name, color) for name, color, *_ in h200],
            panels=[{"title": precision, "categories": WORKLOADS,
                     "values": split(*[row[col][i] for row in h200]),
                     "missing": "TensorRT-LLM NVFP4 needs Blackwell"}
                    for i, precision in enumerate(("BF16", "FP8", "NVFP4 · Marlin W4A16"))],
            xlabel="End-to-end output tokens per second · same scale in all panels",
            legend_columns=3,
            description=f"Stock SGLang, TensorRT-LLM, stock vLLM and the AX tree/chain on one H200 at {clients}, "
                        "for BF16, FP8 and NVFP4.",
        )

    # README: NVFP4 only, the same-pod prefill-graph run (results/h200_prefill_graph).
    bar_chart(
        OUT / "h200-27b-nvfp4-throughput.svg",
        title="H200 · Qwen3.8-27B NVFP4 · one pod",
        subtitle="128 greedy output tokens per request · 8 prompts at C1, 24 at C8 per workload · "
                 "AX five servers, vLLM two",
        series=[("Stock SGLang DFlash2 (Marlin)", SGLANG), ("Stock vLLM DFlash2", VLLM), ("AX tree 32", AX),
                ("AX GDN replay", AX_DARK)],
        panels=[
            {"title": "1 concurrent client", "categories": WORKLOADS,
             "values": split("485.4 / 407.4 / 259.9", "486.3 / 431.5 / 282.8", "587.0 / 483.6 / 294.2",
                             "463.2 / 396.0 / 252.0")},
            {"title": "8 concurrent clients", "categories": WORKLOADS,
             "values": split("1,514.0 / 1,387.4 / 876.1", "1,570.1 / 1,574.0 / 1,063.5",
                             "1,714.0 / 1,720.7 / 1,074.9", "1,658.4 / 1,687.0 / 1,075.7")},
        ],
        xlabel="End-to-end output tokens per second · same scale in both panels",
        note="AX lanes run the NVFP4 linears on Humming and admit one DFlash prefill at a time (launch defaults). "
             "TensorRT-LLM NVFP4 needs Blackwell.",
        description="On one H200 both AX routes lead stock vLLM DFlash2 on every workload at eight clients, and "
                    "the AX tree leads at one client.",
    )

    bar_chart(
        OUT / "rtx-pro-6000-27b-fp8-verify-throughput.svg",
        title="RTX PRO 6000 · Qwen3.8-27B NVFP4",
        subtitle="DFlash2 GDN replay · 8 concurrent requests · 64 chat prompts × 512 tokens",
        series=[("Default FP8 tactics", AX), ("Candidate FP8 tactics", AX_DARK)],
        panels=[{"title": "Default first", "categories": ["Chat"], "values": [[699.74], [718.01]]},
                {"title": "Candidate first", "categories": ["Chat"], "values": [[710.20], [724.04]]}],
        xlabel="End-to-end output tokens per second",
        note="Complete greedy output matched on 57/64 prompts in each order.",
        fmt=lambda v: f"{v:,.2f}",
        description="End-to-end throughput across both launch orders for native GDN replay and replay with the same "
                    "GPU's qualified FP8 tactics.",
    )

    bar_chart(
        OUT / "rtx-27b-nvfp4-throughput.svg",
        title="RTX PRO 6000 · Qwen3.8-27B NVFP4 · earlier pooled sweep",
        subtitle="Ten frozen prompts · 128-token outputs · rates pooled across workloads · SGLang 0.5.23",
        series=[("Stock SGLang AR", SGLANG_LIGHT), ("Stock DFlash2 block 8", SGLANG), ("SGLang + AX Engine", AX_DARK)],
        panels=[{"title": "C1 · 10 requests", "categories": ["All workloads"],
                 "values": [[54.41], [124.50], [156.87]], "suffix": [[None], [None], ["tree 64"]]},
                {"title": "C8 · 40 requests", "categories": ["All workloads"],
                 "values": [[388.29], [489.86], [484.78]],
                 "suffix": [[None], [None], ["tree 32 / chain 8"]]}],
        xlabel="End-to-end output tokens per second", share_scale=False, fmt=lambda v: f"{v:,.2f}",
        description="At concurrency one, AR 54.41, stock DFlash2 124.50 and AX Engine 156.87 output tokens/s. "
                    "At concurrency eight, 388.29, 489.86 and 484.78.",
    )

    bar_chart(
        OUT / "rtx-pro-6000-27b-batched-trees.svg",
        title="RTX PRO 6000 · Qwen3.8-27B NVFP4 · batched lattice trees",
        subtitle="200 prompts · 256 greedy output tokens · 8 slots · 16-node trees · fused GDN kernels",
        series=[("AX tree 16, chain rounds above one request", AX_LIGHT), ("AX batched trees (AX_BTREE=4)", AX),
                ("AX batched trees + AX_TREE_TEMP=1.55", AX_DARK)],
        panels=[{"title": "C1", "categories": ["Output"], "values": [[235.1], [235.3], [244.1]]},
                {"title": "C2", "categories": ["Output"], "values": [[336.6], [395.1], [407.7]]},
                {"title": "C4", "categories": ["Output"], "values": [[536.8], [678.8], [689.0]]}],
        xlabel="Output tokens per second", legend_columns=2, share_scale=False,
        description="Batched trees gain 17% at two requests and 26% at four over chain rounds; the tree temperature "
                    "adds about 3%.",
    )

    bar_chart(
        OUT / "rtx-pro-6000-flash-next-nvfp4-throughput.svg",
        title="RTX PRO 6000 · Qwen3.8-Flash-Next NVFP4",
        subtitle="128 greedy output tokens per request · two passes per route · 8 prompts at C1, 24 at C8 per workload",
        series=[("Stock AR", SGLANG_LIGHT), ("Stock MTP3", SGLANG), ("AX MTP3", AX), ("AX MTP7", AX_DARK)],
        panels=[
            {"title": "1 concurrent client", "categories": WORKLOADS,
             "values": split("83.1 / 83.8 / 83.7", "183.5 / 164.1 / 134.0", "182.1 / 164.8 / 135.0",
                             "218.9 / 174.0 / 114.7")},
            {"title": "8 concurrent clients", "categories": WORKLOADS,
             "values": split("414.1 / 421.5 / 399.3", "679.0 / 582.6 / 455.4", "594.2 / 572.4 / 405.5",
                             "728.0 / 542.7 / 356.5")},
        ],
        xlabel="End-to-end output tokens per second", note="All routes used pinned-host FP8 PLE.",
        share_scale=False,
        description="End-to-end throughput at one and eight concurrent clients by workload and serving route.",
    )

    bar_chart(
        OUT / "rtx-5090-27b-nvfp4-throughput.svg",
        title="GeForce RTX 5090 · Qwen3.8-27B NVFP4",
        subtitle="SGLang 0.5.23 · 128 greedy output tokens per request · two passes · two GPU request slots",
        series=[("Stock SGLang AR", SGLANG_LIGHT), ("Stock DFlash2 block 8", SGLANG), ("SGLang + AX Engine", AX_DARK)],
        panels=[{"title": "C1 · 10 requests", "categories": ["All workloads"],
                 "values": [[57.64], [125.01], [135.18]], "suffix": [[None], [None], ["tree 32"]]},
                {"title": "C2 · 40 requests", "categories": ["All workloads"],
                 "values": [[101.77], [182.38], [181.36]],
                 "suffix": [[None], [None], ["tree 32 / chain 8"]]}],
        xlabel="End-to-end output tokens per second", share_scale=False, fmt=lambda v: f"{v:,.2f}",
        description="Concurrency 1: stock AR 57.64, stock DFlash2 125.01, AX Engine 135.18. Concurrency 2: 101.77, "
                    "182.38 and 181.36 output tokens per second.",
    )

    bar_chart(
        OUT / "rtx-5090-27b-batched-trees.svg",
        title="GeForce RTX 5090 · Qwen3.8-27B NVFP4 · trees at low concurrency",
        subtitle="200 prompts · 256 greedy output tokens · 2 slots · AX_SSM_POSITIONS=8 · AX_TREE_TEMP=1.55",
        series=[("AX GDN replay, DFlash2 chain 16", AX_LIGHT), ("AX tree 16, chain 8 above one request", AX),
                ("AX tree 32", AX_MID), ("AX batched trees (AX_BTREE=2)", AX_DARK)],
        panels=[{"title": "C1", "categories": ["Output"],
                 "values": [[187.6], [220.4], [237.8], [220.2]]},
                {"title": "C2", "categories": ["Output"],
                 "values": [[338.5], [307.7], [310.6], [370.6]]}],
        xlabel="Output tokens per second", legend_columns=2, share_scale=False,
        description="At one request the 32-node tree leads; at two requests batched trees lead by 9.5% over GDN "
                    "replay and 20.4% over chain rounds.",
    )

    bar_chart(
        OUT / "rtx-5090-27b-gdn-replay-throughput.svg",
        title="GeForce RTX 5090 · DFlash2 GDN replay",
        subtitle="Qwen3.8-27B NVFP4 · width-8 chain · 128 output tokens per request",
        series=[("Stock DFlash2", SGLANG), ("AX GDN replay", AX_DARK)],
        panels=[{"title": "C1 · 2 slots", "categories": ["Output"], "values": [[123.21], [124.39]]},
                {"title": "C2 · 2 slots", "categories": ["Output"], "values": [[178.67], [181.19]]},
                {"title": "C8 · 8 slots", "categories": ["Output"], "values": [[None], [389.63]],
                 "missing": "stock launch exceeded GPU memory"}],
        xlabel="Output tokens per second", share_scale=False, fmt=lambda v: f"{v:,.2f}", legend_columns=2,
        description="Concurrency 1, stock 123.21 and replay 124.39. Concurrency 2, stock 178.67 and replay 181.19. "
                    "Concurrency 8, replay 389.63. The stock launch exceeded GPU memory.",
    )

    bar_chart(
        OUT / "rtx-5090-27b-fp8-verify-throughput.svg",
        title="GeForce RTX 5090 · Qwen3.8-27B NVFP4",
        subtitle="DFlash2 GDN replay · 8 concurrent requests · 64 chat prompts × 512 tokens",
        series=[("Default FP8 tactics", AX), ("Candidate FP8 tactics", AX_DARK)],
        panels=[{"title": "Default first", "categories": ["Chat"], "values": [[802.33], [824.01]]},
                {"title": "Candidate first", "categories": ["Chat"], "values": [[807.25], [824.02]]}],
        xlabel="End-to-end output tokens per second",
        note="Complete greedy output matched on 10/64 prompts in each order.",
        fmt=lambda v: f"{v:,.2f}",
        description="End-to-end throughput across both launch orders for native GDN replay and replay with the same "
                    "GPU's qualified FP8 tactics.",
    )

    thor = [("Stock SGLang AR", SGLANG_LIGHT, "13.4 / 13.5 / 13.5", "4.6 / 4.6 / 4.6"),
            ("Edge-LLM AR", EDGE_LIGHT, "13.3 / 13.3 / 13.3", "4.7 / 4.7 / 4.7"),
            ("Stock SGLang MTP3", SGLANG, "36.2 / 35.2 / 27.0", "13.5 / 12.9 / 9.9"),
            ("Edge-LLM MTP3", EDGE, "39.1 / 37.3 / 28.4", "13.8 / 12.9 / 10.2"),
            ("Edge-LLM DFlash2 chain 8", EDGE_MID, "73.0 / 65.7 / 34.5", "30.1 / 25.1 / 14.0"),
            ("Edge-LLM DDTree 32", EDGE_DARK, "71.4 / 68.8 / 44.2", "30.1 / 28.8 / 18.2"),
            ("Stock vLLM DFlash2 chain 8", VLLM, "65.9 / 57.6 / 33.1", "28.7 / 24.4 / 13.6"),
            ("Stock SGLang DFlash2 chain 16", SGLANG_DARK, "93.0 / 73.5 / 30.6", "43.1 / 29.6 / 12.5"),
            ("AX Engine tree 32", AX_DARK, "106.1 / 92.1 / 43.1", "47.5 / 37.6 / 18.3")]
    bar_chart(
        OUT / "jetson-thor-27b-throughput.svg",
        title="Jetson Thor · Qwen3.8-27B",
        subtitle="Concurrency 1 · arm64 · SGLang 0.5.23, vLLM 0.34.0rc1 and TensorRT Edge-LLM 0.12.0 · same checkpoint, drafter, prompts",
        series=[(name, color) for name, color, *_ in thor],
        panels=[{"title": "BF16 · 8 prompts · 384-token limit", "categories": WORKLOADS,
                 "values": split(*[row[3] for row in thor])},
                {"title": "NVFP4 · 12 prompts · 512-token limit", "categories": WORKLOADS,
                 "values": split(*[row[2] for row in thor])}],
        xlabel="Output tokens per second · each precision has its own scale", share_scale=False,
        description="Stock SGLang, stock vLLM, TensorRT Edge-LLM and the add-on, grouped by BF16 or NVFP4 and workload.",
    )

    bar_chart(
        OUT / "jetson-thor-27b-nvfp4-throughput.svg",
        title="Jetson Thor · Qwen3.8-27B NVFP4",
        subtitle="Concurrency 1 · arm64 · SGLang 0.5.23, vLLM 0.34.0rc1 and TensorRT Edge-LLM 0.12.0 · "
                 "12 prompts · 512-token limit",
        series=[(name, color) for name, color, *_ in thor],
        panels=[{"title": "NVFP4", "categories": WORKLOADS, "values": split(*[row[2] for row in thor])}],
        xlabel="Output tokens per second",
        description="Stock SGLang, stock vLLM, TensorRT Edge-LLM and the add-on on one Jetson Thor for NVFP4. "
                    "The AX tree leads code and math and ties Edge-LLM DDTree 32 on chat.",
    )

    bar_chart(
        OUT / "thor-01-fp8-projection-time.svg",
        title="Thor-01 · FP8 projection time reduction",
        subtitle="Synthetic CUDA graph benchmark vs. default cuBLAS · bit-exact BF16",
        series=[("GDN QKVZ", AX), ("Attention QKV", AX_DARK)],
        panels=[{"title": "48 rows", "categories": ["Reduction"], "values": [[4.65], [4.55]]},
                {"title": "64 rows", "categories": ["Reduction"], "values": [[4.49], [5.88]]},
                {"title": "96 rows", "categories": ["Reduction"], "values": [[0.62], [1.70]]},
                {"title": "128 rows", "categories": ["Reduction"], "values": [[0.84], [1.07]]}],
        share_scale=False,
        xlabel="Projection time reduction (%)", fmt=lambda v: f"{v:.2f}%", legend_columns=2, tick_suffix="%",
        description="Bit-exact projection times on synthetic CUDA inputs at 48, 64, 96 and 128 rows, relative to "
                    "default cuBLAS.",
    )

    bar_chart(
        OUT / "dflash-draft-vocab-retest.svg",
        title="Qwen3.8-27B NVFP4 · DFlash2 draft vocabulary subset",
        subtitle="Throughput change with the subset enabled, by launch order · fixed-code retest",
        series=[("Full head first", AX), ("Subset first", AX_DARK)],
        panels=[{"title": "Thor-01", "categories": ["C1", "C2"],
                 "values": [[2.19, 2.33], [-0.29, 0.05]]},
                {"title": "B200", "categories": ["C1", "C8"],
                 "values": [[-1.07, -0.34], [-2.47, -1.47]]}],
        xlabel="Change in output tokens per second vs. the full draft head (%)", fmt=percent, legend_columns=2, tick_suffix="%",
        note="Thor-01 full-first gains came from a slow first server. B200 lanes are untimed; servers varied by up to 2.6%.",
        description="Throughput change from the draft vocabulary subset on Thor-01 and B200 in both launch orders.",
    )

    single = [("Edge-LLM 0.12.0 MTP3", EDGE, "37.2 / 34.7 / 26.4", "89.8 / 90.3 / 67.0"),
              ("Stock SGLang MTP3", SGLANG, "42.3 / 38.8 / 28.9", "96.2 / 92.2 / 65.4"),
              ("Stock vLLM MTP3", VLLM, "42.4 / 41.2 / 31.3", "101.0 / 101.5 / 75.8"),
              ("AX SGLang adaptive MTP + 64K draft vocabulary", AX, "59.8 / 49.1 / 33.6", "123.3 / 103.1 / 70.5"),
              ("AX vLLM adaptive MTP + NVFP4 draft head", AX_DARK, "62.4 / 53.3 / 36.9", "131.4 / 117.2 / 82.7")]
    bar_chart(
        OUT / "jetson-thor-single-flash-next-adaptive-mtp.svg",
        title="One Jetson Thor · Qwen3.8-Flash-Next NVFP4",
        subtitle="Greedy serving · 48 prompts per workload · 256-token limit · PLE table on local NVMe",
        series=[(name, color) for name, color, *_ in single],
        panels=[{"title": "1 concurrent request", "categories": WORKLOADS, "values": split(*[r[2] for r in single])},
                {"title": "4 concurrent requests", "categories": WORKLOADS, "values": split(*[r[3] for r in single])}],
        xlabel="End-to-end output tokens per second", legend_columns=2, share_scale=False,
        description="Stock MTP on SGLang, vLLM and TensorRT Edge-LLM, and adaptive MTP with the add-ons.",
    )

    bar_chart(
        OUT / "jetson-thor-flash-next-mtp-throughput.svg",
        title="Two Jetson Thor hosts · Qwen3.8-Flash-Next NVFP4",
        subtitle="Concurrency 1 · 24 prompts per workload · 256-token cap · two pipeline stages over 2.5 GbE",
        series=[("Stock SGLang AR", SGLANG_LIGHT), ("Stock SGLang MTP3", SGLANG), ("MTP3 + AX QSA fix", AX),
                ("MTP7 + AX QSA fix", AX_DARK)],
        panels=[{"categories": WORKLOADS,
                 "values": split("18.3 / 19.0 / 19.2", "32.0 / 32.4 / 24.8", "33.3 / 32.5 / 25.7",
                                 "43.1 / 37.5 / 22.8")}],
        xlabel="Output tokens per second",
        description="MTP7 with the QSA fix leads on code and math. MTP3 with the fix leads on chat.",
    )

    bar_chart(
        OUT / "thor-01-dflash-draft-vocab.svg",
        title="Thor-01 · DFlash2 draft vocabulary",
        subtitle="Qwen3.8-27B NVFP4 weight projection and top-k on synthetic inputs",
        series=[("Full · 248,320 rows", AX), ("Subset · 148,736 rows", AX_DARK)],
        panels=[{"categories": ["7", "14", "56"],
                 "values": [[3.738, 2.898, 3.340], [2.224, 1.744, 1.999]]}],
        xlabel="Median CUDA graph time (ms) · draft rows",
        fmt=lambda v: f"{v:.3f}", legend_columns=2,
        description="Median CUDA graph time for the full and reduced draft vocabulary projection and top-k at 7, 14 "
                    "and 56 draft rows.",
    )

    bar_chart(
        OUT / "dflash-draft-vocab-serving.svg",
        title="Qwen3.8-27B NVFP4 · DFlash2 draft vocabulary",
        subtitle="Throughput change with the subset enabled, by launch order",
        series=[("Full head first", AX), ("Subset first", AX_DARK)],
        panels=[
            {"title": "Slot refill · 16 requests",
             "categories": ["Thor-01 · C1", "Thor-01 · C2", "RTX PRO 6000 · C1", "RTX PRO 6000 · C8",
                            "B200 · C1", "B200 · C8"],
             "values": [[0.86, -0.02, 0.43, 0.99, 6.52, 2.46], [0.14, -0.10, 1.75, -0.71, 0.47, -14.03]]},
            {"title": "Fixed groups · 80 requests", "categories": ["B200 · C8"],
             "values": [[-0.38], [-0.09]]},
        ],
        xlabel="Change in output tokens per second vs. the full draft head (%)",
        fmt=percent, legend_columns=2, tick_suffix="%",
        description="Throughput change from the draft vocabulary subset on Thor-01, RTX PRO 6000 and B200 in both "
                    "launch orders, plus the fixed-group B200 retest.",
    )


if __name__ == "__main__":
    main()
