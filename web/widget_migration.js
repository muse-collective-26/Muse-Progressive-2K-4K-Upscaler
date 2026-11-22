import { app } from "../../scripts/app.js";

// Repair the first published workflow, which omitted the seed control widget.
app.registerExtension({
  name: "Muse.ProgressiveUpscaler.WidgetMigration",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MuseProgressiveUpscaler") return;
    const original = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      original?.apply(this, arguments);
      const v = info.widgets_values;
      if (Array.isArray(v) && typeof v[3] === "string" && v[3].endsWith(".safetensors")) {
        const restored = {
          output: v[0], strength: v[1], seed: v[2], control_after_generate: "fixed",
          base_model: v[3], text_encoder: v[4], video_vae: v[5],
          audio_vae: v[6], regenerate_lora: v[7],
        };
        for (const w of this.widgets ?? []) {
          if (Object.hasOwn(restored, w.name)) w.value = restored[w.name];
        }
      }
      const prompt = this.widgets?.find(w => w.name === "preservation_prompt");
      if (prompt && typeof prompt.value === "string" && prompt.value.startsWith("subject_definitions:")) {
        prompt.value = nodeData.input.required.preservation_prompt[1].default;
      }
      this.title = "Muse Progressive 2K / 4K Upscaler";
    };
  },
});
