import gradio as ui
import subprocess
import os
import sys


def run_pipeline():
    logs = []

    current_python = sys.executable

    # logs.append("=== STAGE 1: Starting Data Generation ===")
    # yield "\n".join(logs), None
    #
    # process = subprocess.Popen([current_python, "data/generate_hdf5.py"], stdout=subprocess.PIPE,
    #                            stderr=subprocess.STDOUT, text=True)
    # for line in process.stdout:
    #     logs.append(f"[DataGen] {line.strip()}")
    #     yield "\n".join(logs), None
    # process.wait()

    logs.append("\n=== STAGE 2: Starting Model Training (5 Epochs for Demo) ===")
    yield "\n".join(logs), None

    process = subprocess.Popen([current_python, "train.py", "--epochs", "5", "--batch_size", "16"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        logs.append(f"[Train] {line.strip()}")
        yield "\n".join(logs), None
    process.wait()

    logs.append("\n=== STAGE 3: Running Test Set Evaluation ===")
    yield "\n".join(logs), None

    process = subprocess.Popen([current_python, "evaluate.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
    for line in process.stdout:
        logs.append(f"[Eval] {line.strip()}")
        yield "\n".join(logs), None
    process.wait()

    logs.append("\n=== STAGE 4: Generating Prediction Plots ===")
    yield "\n".join(logs), None

    process = subprocess.Popen([current_python, "-m", "utils.visualizer"], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    process.wait()

    logs.append("\nPipeline Completed Successfully!")

    plot_path = "plots/restoration_grid.png"
    if os.path.exists(plot_path):
        yield "\n".join(logs), plot_path
    else:
        yield "\n".join(logs), None

# --- Gradio UI Layout Interface ---
with ui.Blocks(title="OAM Turbulence Restorer Dashboard") as demo:
    ui.Markdown("# 🌌 OAM Turbulence Compensation Control Panel")
    ui.Markdown(
        "Click the button below to sequentially execute data generation, model training, evaluation, and plotting maps.")

    with ui.Row():
        with ui.Column(scale=1):
            run_btn = ui.Button("🚀 Run Sequential Pipeline", variant="primary")
            output_logs = ui.Textbox(label="Live Terminal Execution Logs", lines=20, max_lines=25, autoscroll=True)

        with ui.Column(scale=1):
            ui.Markdown("### Generated Restoration Visualizations")
            output_image = ui.Image(label="Restoration Grid (Clean vs Distorted vs Predicted)")

    # Connect button trigger to yield live text streams and the final image output
    run_btn.click(fn=run_pipeline, inputs=None, outputs=[output_logs, output_image])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)