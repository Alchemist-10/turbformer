import gradio as gr
import subprocess
import sys
import os


DATA_FILES = ["train.h5", "val.h5", "test_ood.h5"]


def run_command(cmd, stage_name):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        yield f"[{stage_name}] {line.rstrip()}"

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"{stage_name} failed with exit code {process.returncode}")


def run_pipeline(generate_data, epochs, batch_size):
    logs = []

    python_exec = sys.executable

    try:
        # =====================================================
        # Stage 1 - Data Generation
        # =====================================================
        if generate_data:
            logs.append("=== STAGE 1: DATA GENERATION ===")
            yield "\n".join(logs), None

            for line in run_command(
                [python_exec, "data/generate_hdf5.py"],
                "DataGen"
            ):
                logs.append(line)
                yield "\n".join(logs), None

        else:
            missing = [f for f in DATA_FILES if not os.path.exists(f)]

            if missing:
                logs.append(
                    f"Dataset files missing: {missing}. Generating automatically."
                )
                yield "\n".join(logs), None

                for line in run_command(
                    [python_exec, "data/generate_hdf5.py"],
                    "DataGen"
                ):
                    logs.append(line)
                    yield "\n".join(logs), None
            else:
                logs.append("Dataset already exists. Skipping generation.")
                yield "\n".join(logs), None

        # =====================================================
        # Stage 2 - Training
        # =====================================================
        logs.append("\n=== STAGE 2: TRAINING ===")
        yield "\n".join(logs), None

        for line in run_command(
            [
                python_exec,
                "train.py",
                "--epochs",
                str(epochs),
                "--batch_size",
                str(batch_size),
            ],
            "Train",
        ):
            logs.append(line)
            yield "\n".join(logs), None

        # =====================================================
        # Stage 3 - Evaluation
        # =====================================================
        logs.append("\n=== STAGE 3: EVALUATION ===")
        yield "\n".join(logs), None

        for line in run_command(
            [python_exec, "evaluate.py"],
            "Eval",
        ):
            logs.append(line)
            yield "\n".join(logs), None

        # =====================================================
        # Stage 4 - Visualization
        # =====================================================
        logs.append("\n=== STAGE 4: VISUALIZATION ===")
        yield "\n".join(logs), None

        if os.path.exists("utils/visualizer.py"):
            for line in run_command(
                [python_exec, "-m", "utils.visualizer"],
                "Visualizer",
            ):
                logs.append(line)
                yield "\n".join(logs), None

        logs.append("\n✅ PIPELINE COMPLETED SUCCESSFULLY")

        plot_candidates = [
            "plots/restoration_grid.png",
            "outputs/restoration_grid.png",
            "results/restoration_grid.png",
        ]

        image_path = None

        for path in plot_candidates:
            if os.path.exists(path):
                image_path = path
                break

        yield "\n".join(logs), image_path

    except Exception as e:
        logs.append(f"\n❌ ERROR: {str(e)}")
        yield "\n".join(logs), None


with gr.Blocks(title="TurbFormer Dashboard") as demo:
    gr.Markdown("# 🌌 TurbFormer Dashboard")
    gr.Markdown(
        "Train, evaluate, and visualize the atmospheric turbulence compensation pipeline."
    )

    with gr.Row():
        with gr.Column(scale=1):
            generate_checkbox = gr.Checkbox(
                label="Regenerate Dataset",
                value=False,
            )

            epochs_input = gr.Slider(
                minimum=1,
                maximum=100,
                value=10,
                step=1,
                label="Epochs",
            )

            batch_input = gr.Slider(
                minimum=1,
                maximum=32,
                value=8,
                step=1,
                label="Batch Size",
            )

            run_button = gr.Button(
                "🚀 Run Pipeline",
                variant="primary",
            )

            logs_output = gr.Textbox(
                label="Live Logs",
                lines=25,
                autoscroll=True,
            )

        with gr.Column(scale=1):
            image_output = gr.Image(
                label="Restoration Visualization"
            )

    run_button.click(
        fn=run_pipeline,
        inputs=[
            generate_checkbox,
            epochs_input,
            batch_input,
        ],
        outputs=[
            logs_output,
            image_output,
        ],
    )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)