from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def generate_layout_data(dsp_x, bram_x, req_io, req_clb, io_cap=8):
    """
    Generates legal fixed-layout data for VTR.

    Returns:
        width, height, dsps, mems
    """

    column_heights = {}
    dsps = []
    mems = []

    # DSP height is 4
    for coord in dsp_x:
        x = max(1, int(round(coord)))
        y = column_heights.get(x, 1)

        dsps.append((x, y))
        column_heights[x] = y + 4

    # BRAM height is 6
    for coord in bram_x:
        x = max(1, int(round(coord)))
        y = column_heights.get(x, 1)

        mems.append((x, y))
        column_heights[x] = y + 6

    placed_blocks = []

    for x, y in dsps:
        placed_blocks.append(("mult_36", x, y, 4))

    for x, y in mems:
        placed_blocks.append(("memory", x, y, 6))

    if placed_blocks:
        max_used_x = max(block[1] for block in placed_blocks)
        max_used_y = max(block[2] + block[3] - 1 for block in placed_blocks)
    else:
        max_used_x = 1
        max_used_y = 1

    # Add perimeter space
    width = max_used_x + 2
    height = max_used_y + 2

    # Expand until IO and CLB requirements are satisfied
    while True:
        io_tiles = 2 * (width - 2) + 2 * (height ) -4
        curr_io = io_tiles * io_cap

        core_area = (width - 2) * (height - 2)
        dsp_tiles = len(dsps) * 4
        bram_tiles = len(mems) * 6
        curr_clb = core_area - dsp_tiles - bram_tiles

        if curr_io >= req_io and curr_clb >= req_clb:
            break

        if width <= height:
            width += 1
        else:
            height += 1

    return width, height, dsps, mems


def bake_architecture(template_path, output_path, width, height, dsps, mems):
    """
    Renders the Jinja architecture template into a real VTR architecture XML file.
    """

    template_path = Path(template_path)
    output_path = Path(output_path)

    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(template_path.name)

    rendered_xml = template.render(
        layout_name="my_layout",
        width=width,
        height=height,
        dsps=dsps,
        mems=mems,
    )

    output_path.write_text(rendered_xml)

    print(f"Successfully baked architecture: {output_path}")


if __name__ == "__main__":
    # Inputs from RL agent
    dsp_x_actions = [3, 3, 5, 10, 10]
    bram_x_actions = [2, 8]

    width, height, dsps, mems = generate_layout_data(
        dsp_x=dsp_x_actions,
        bram_x=bram_x_actions,
    )

    bake_architecture(
        template_path="/root/desktop/vtr_exp/arch/k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml.j2",
        output_path="vpr_arch_run.xml",
        width=width,
        height=height,
        dsps=dsps,
        mems=mems,
    )
