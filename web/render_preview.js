import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const panes = new Map();
api.addEventListener("muse_progressive_preview", ({ detail }) => {
  const pane = panes.get(String(detail.node_id));
  if (!pane) return;
  pane.status.textContent = detail.stage;
  if (detail.image) {
    const params = new URLSearchParams({ ...detail.image, t: String(Date.now()) });
    pane.image.src = api.apiURL(`/view?${params}`);
    pane.image.style.display = "block";
    pane.caption.textContent = detail.stage.startsWith("Starting")
      ? "Source frame - output preview will replace this when 2K completes"
      : "Latest completed output frame";
  }
});

app.registerExtension({
  name: "Muse.ProgressiveUpscaler.RenderPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MuseProgressiveUpscaler") return;
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      configured?.apply(this, arguments);
      this.setSize([Math.max(this.size[0], 520), Math.max(this.size[1], 760)]);
    };
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const root = document.createElement("div");
      Object.assign(root.style, { background: "#11151d", border: "1px solid #41485a",
        borderRadius: "6px", display: "flex", flexDirection: "column", overflow: "hidden",
        width: "100%", height: "280px", color: "#ddd", font: "12px sans-serif" });
      const status = document.createElement("div");
      status.textContent = "Preview - waiting for render";
      Object.assign(status.style, { padding: "8px", background: "#222938" });
      const image = document.createElement("img");
      Object.assign(image.style, { width: "100%", height: "220px", objectFit: "contain", display: "none" });
      const caption = document.createElement("div");
      caption.textContent = "Shows the source, then completed 2K and 4K previews with live stage progress.";
      Object.assign(caption.style, { padding: "6px", fontSize: "11px", color: "#aab2c4" });
      root.append(status, image, caption);
      const widget = this.addDOMWidget("muse_render_preview", "div", root, { serialize: false });
      widget.computeSize = () => [this.size[0], 280];
      const pane = { status, image, caption };
      const added = this.onAdded;
      this.onAdded = function () {
        added?.apply(this, arguments);
        panes.set(String(this.id), pane);
      };
      panes.set(String(this.id), pane);
      const removed = this.onRemoved;
      this.onRemoved = function () {
        panes.delete(String(this.id));
        root.remove();
        removed?.apply(this, arguments);
      };
      this.setSize([Math.max(this.size[0], 520), Math.max(this.size[1], 760)]);
    };
  },
});
