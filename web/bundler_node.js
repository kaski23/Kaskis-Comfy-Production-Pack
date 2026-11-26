import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "KASKI.Bundler",

    registerCustomNodes() {
        const { LiteGraph, LGraphNode } = window;

        class Bundler extends LGraphNode {
            static title = "Bundler";
            static description = "Route multiple independent connections through one node.";

            constructor(title) {
                super(title);
                // ComfyUI resolves each output through the input at the same index.
                this.isVirtualNode = true;
                this.updateChannels(1);
            }

            updateChannels(count) {
                if (this.inputs.length === count && this.outputs.length === count) return;

                // Only remove trailing slots, so existing links keep their indices.
                while (this.inputs.length > count) this.removeInput(this.inputs.length - 1);
                while (this.outputs.length > count) this.removeOutput(this.outputs.length - 1);

                for (let slot = this.inputs.length; slot < count; slot++) {
                    this.addInput(`input_${slot + 1}`, "*");
                }
                for (let slot = this.outputs.length; slot < count; slot++) {
                    this.addOutput(`output_${slot + 1}`, "*");
                }

                const [width, height] = this.computeSize();
                this.setSize([Math.max(this.size[0], width), height]);
                this.setDirtyCanvas(true, true);
            }

            syncChannels() {
                let lastUsed = Math.max(this.inputs.length, this.outputs.length) - 1;
                while (lastUsed >= 0) {
                    if (this.isInputConnected(lastUsed) || this.isOutputConnected(lastUsed)) break;
                    lastUsed--;
                }
                this.updateChannels(lastUsed + 2);
            }

            onConnectionsChange(_type, _slot, connected, link) {
                if (!this.graph || this.configuringChannels || app.configuringGraph) return;
                if (connected) {
                    if (link) this.syncChannels();
                } else {
                    // Allow replacements and graph edits to settle before trimming.
                    requestAnimationFrame(() => {
                        if (this.graph && !app.configuringGraph) this.syncChannels();
                    });
                }
            }

            configure(info) {
                this.configuringChannels = true;
                try {
                    super.configure(info);
                    // Saved slot indices must exist before the graph restores its links.
                    this.updateChannels(Math.max(1, this.inputs.length, this.outputs.length));
                } finally {
                    this.configuringChannels = false;
                }
            }
        }

        LiteGraph.registerNodeType("Bundler_KASKI", Bundler);
        Bundler.category = "KASKI/flow_control";
    },

    beforeRegisterVueAppNodeDefs(defs) {
        const definition = defs.find((node) => node.name === "Bundler_KASKI");
        if (definition) definition.display_name = "Bundler";
    },
});
