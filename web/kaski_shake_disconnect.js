import { app } from "../../scripts/app.js";

// Shake-Einstellungen
const SHAKE_DISTANCE_PX = 12;   // Mindestweg pro Bewegung in Bildschirmpixeln. Kleiner = empfindlicher.
const SHAKE_REVERSALS = 3;      // Anzahl der Richtungswechsel. 3 entspricht rechts–links–rechts–links.
const SHAKE_WINDOW_MS = 650;    // Zeitfenster in Millisekunden. Größer = mehr Zeit zum Schütteln.

const requiredStrokes = SHAKE_REVERSALS + 1;

function newAxis(position) {
    return { extreme: position, direction: 0, strokes: [] };
}

function recordStroke(axis, position, time) {
    const delta = position - axis.extreme;
    const direction = Math.sign(delta);
    if (direction === axis.direction) {
        axis.extreme = position;
        return false;
    }
    if (Math.abs(delta) < SHAKE_DISTANCE_PX) return false;

    axis.direction = direction;
    axis.extreme = position;
    axis.strokes.push(time);
    if (axis.strokes.length > requiredStrokes) axis.strokes.shift();
    return axis.strokes.length === requiredStrokes && time - axis.strokes[0] <= SHAKE_WINDOW_MS;
}

function getLink(graph, id) {
    return graph.getLink?.(id) ?? graph.links?.get?.(id) ?? graph.links?.[id];
}

function endpoint(graph, id, index, side) {
    const boundary = side === "outputs" ? graph.inputNode : graph.outputNode;
    const node = boundary?.id === id ? boundary : graph.getNodeById(id);
    if (!node) return;
    const slotsKey = node === boundary ? "slots" : side;
    const slot = node[slotsKey]?.[index];
    if (slot) return { node, slot, slotsKey };
}

function liveSlot(graph, port) {
    const { node, slot, slotsKey } = port;
    if (slotsKey !== "slots" && graph.getNodeById(node.id) !== node) return;
    const slots = node[slotsKey];
    if (!slots) return;
    if (slots.includes(slot)) return slot;
    // A connection callback may have rebuilt the slots. Rebind by unique name.
    if (!slot.name) return;
    const matches = slots.filter((other) => other?.name === slot.name);
    if (matches.length === 1) return matches[0];
}

function compatible(a, b) {
    return window.LiteGraph.isValidConnection(a, b);
}

function chooseSource(node, sources, output, index, target) {
    const candidates = sources.filter(({ from, carriedType }) =>
        from.node !== target.node && compatible(from.slot.type, target.slot.type) &&
        compatible(carriedType, target.slot.type)
    );
    // The pack's Bundler has independent wildcard channels, including empty ones.
    if (node.type === "Bundler_KASKI") return candidates.find((s) => s.index === index);

    const through = candidates.filter((s) =>
        compatible(s.input.type, output.type) && compatible(s.carriedType, output.type)
    );
    const pool = through.length ? through : candidates;
    const outputName = output.name?.trim().toLowerCase();
    const named = outputName && pool.filter((s) => s.input.name?.trim().toLowerCase() === outputName);
    if (named?.length === 1) return named[0];
    const sameIndex = pool.find((s) => s.index === index);
    if (sameIndex) return sameIndex;
    const first = pool[0];
    if (first && pool.every((s) => s.from.node === first.from.node && s.from.slot === first.from.slot)) {
        return first;
    }
}

function planBypass(graph, node) {
    const ids = new Set();
    const sources = [];
    const routes = [];
    for (const [index, input] of (node.inputs ?? []).entries()) {
        if (input?.link == null) continue;
        ids.add(input.link);
        const link = getLink(graph, input.link);
        if (!link || link.origin_id === node.id) continue;
        const from = endpoint(graph, link.origin_id, link.origin_slot, "outputs");
        if (!from) continue;
        const type = from.slot.type;
        const carriedType = !type || type === "*" ? (link.type ?? type) : type;
        sources.push({ index, input, from, carriedType, parentId: link.parentId });
    }
    for (const [index, output] of (node.outputs ?? []).entries()) {
        for (const id of output?.links ?? []) {
            ids.add(id);
            const link = getLink(graph, id);
            if (!link || link.target_id === node.id) continue;
            const to = endpoint(graph, link.target_id, link.target_slot, "inputs");
            if (!to) continue;
            const source = chooseSource(node, sources, output, index, to);
            if (source) routes.push({ from: source.from, to, oldId: id, parentId: source.parentId });
        }
    }
    return { ids, routes };
}

function singleLinkOutput(slot) {
    return slot.type === window.LiteGraph.EVENT && !window.LiteGraph.allow_multi_output_for_events;
}

function connectRoute(graph, { from, to, oldId, parentId }) {
    const output = liveSlot(graph, from);
    const input = liveSlot(graph, to);
    if (!output || !input || !compatible(output.type, input.type)) return;
    if (singleLinkOutput(output) && output.links?.length) return;
    // Another connection callback may have filled this input in the meantime.
    if (to.node === graph.outputNode) {
        if (input.linkIds.some((id) => id !== oldId)) return;
    } else if (input.link != null && input.link !== oldId) {
        return;
    }
    const reroute = parentId != null && graph.getReroute?.(parentId) ? parentId : undefined;
    if (from.node === graph.inputNode) {
        // LiteGraph has no direct public-input to public-output connection.
        if (to.node !== graph.outputNode) output.connect(input, to.node, reroute);
    } else if (to.node === graph.outputNode) {
        input.connect(output, from.node, reroute);
    } else {
        const targetIndex = to.node.inputs.indexOf(input);
        // This route owns one destination socket. Never follow a custom
        // redirection to a different, possibly already wired input.
        if (to.node.onBeforeConnectInput &&
            to.node.onBeforeConnectInput(targetIndex, targetIndex) !== targetIndex) return;
        if (to.node.inputs[targetIndex] !== input ||
            (input.link != null && input.link !== oldId)) return;
        if (typeof from.node.connectSlots === "function") {
            from.node.connectSlots(output, to.node, input, reroute);
        } else {
            from.node.connect(from.node.outputs.indexOf(output), to.node, targetIndex, reroute);
        }
    }
}

function tryConnectRoute(graph, route) {
    try {
        connectRoute(graph, route);
    } catch (error) {
        // Third-party nodes can throw from a connection hook. Finish the
        // detach and the independent branches even if this target refuses.
        console.warn("[KASKI Shake] Could not reconnect", route.from.node.id, "→", route.to.node.id, error);
    }
}

function floatingLinks(graph, node) {
    return Array.from(graph.floatingLinks?.values?.() ?? []).filter((link) =>
        link.origin_id === node.id || link.target_id === node.id
    );
}

function detachNode(canvas, graph, node) {
    // Snapshot before callbacks can remove or reorder dynamic channels.
    const { ids, routes } = planBypass(graph, node);
    if (!ids.size && !floatingLinks(graph, node).length) return;
    canvas.emitBeforeChange?.();
    graph.beforeChange?.();
    try {
        const deferred = [];
        for (const route of routes) {
            // Replace ordinary links in place so dynamic destination inputs stay alive.
            // Boundaries and single-link event outputs must release the old wire first.
            if (route.from.node === graph.inputNode || route.to.node === graph.outputNode ||
                singleLinkOutput(route.from.slot)) deferred.push(route);
            else tryConnectRoute(graph, route);
        }
        for (const id of ids) {
            const link = getLink(graph, id);
            if (!link || (link.origin_id !== node.id && link.target_id !== node.id)) continue;
            const boundary = graph.outputNode;
            const slot = boundary && link.target_id === boundary.id
                ? boundary.slots[link.target_slot] : null;
            if (slot?.linkIds?.includes(id)) slot.disconnect();
            else graph.removeLink(id);
        }
        // Replacement may itself leave a floating reroute on the detached node.
        for (const link of floatingLinks(graph, node)) graph.removeFloatingLink(link);
        for (const route of deferred) tryConnectRoute(graph, route);
    } finally {
        graph.afterChange?.();
        canvas.emitAfterChange?.();
        canvas.setDirty(true, true);
    }
}

function hitNode(canvas, event) {
    const target = event.target;
    if (target === canvas.canvas) {
        const [x, y] = canvas.convertEventToCanvasOffset(event);
        const node = canvas.graph.getNodeOnPos(x, y);
        return node && { node, element: null };
    }

    // Nodes 2.0 draws nodes as HTML siblings of the graph canvas.
    if (!(target instanceof Element) || !canvas.canvas.parentElement?.contains(target)) return;
    if (target.closest('button, a, input, select, textarea, [role="button"], [role="slider"], [contenteditable]:not([contenteditable="false"])')) return;
    const element = target.closest(".lg-node[data-node-id], [data-node-type][data-node-id]");
    if (!element) return;
    const id = element.dataset.nodeId;
    const node = canvas.graph.getNodeById(id) ??
        (/^\d+$/.test(id) ? canvas.graph.getNodeById(Number(id)) : null);
    return node && { node, element };
}

let installed = false;
function installShake() {
    if (installed) return;
    installed = true;
    let drag = null;
    const reset = () => { drag = null; };
    const finish = (event) => {
        if (drag?.pointerId === event.pointerId) reset();
    };

    window.addEventListener("pointerdown", (event) => {
        if (drag && drag.pointerId !== event.pointerId) reset();
        if (drag || !event.isPrimary || event.button !== 0 ||
            event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
        const canvas = app.canvas;
        if (!canvas?.graph || canvas.read_only) return;
        const hit = hitNode(canvas, event);
        if (!hit || hit.node.flags?.pinned) return;
        drag = {
            ...hit, canvas, graph: canvas.graph, pointerId: event.pointerId,
            startX: hit.node.pos[0], startY: hit.node.pos[1],
            startTransform: hit.element?.style.transform, moved: false,
            x: newAxis(event.clientX), y: newAxis(event.clientY),
        };
    }, true);

    window.addEventListener("pointermove", (event) => {
        if (!drag || drag.pointerId !== event.pointerId) return;
        const { node, graph, canvas, element } = drag;
        if (event.buttons !== 1 || canvas !== app.canvas || canvas.graph !== graph ||
            canvas.read_only || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
            reset();
            return;
        }

        // Pan, link drags and widget edits can all move the pointer over a node.
        // Count a shake only if the node itself has moved in graph space.
        if (node.pos[0] !== drag.startX || node.pos[1] !== drag.startY ||
            (element && element.style.transform !== drag.startTransform)) {
            drag.moved = true;
        }
        const horizontal = recordStroke(drag.x, event.clientX, event.timeStamp);
        const vertical = recordStroke(drag.y, event.clientY, event.timeStamp);
        if (!(horizontal || vertical) || !drag.moved) return;
        if (!element && !(canvas.isDragging || canvas.node_dragged === node)) return;
        if (graph.getNodeById(node.id) !== node) return;
        reset(); // Only once per drag, even if a custom node immediately creates new links.
        detachNode(canvas, graph, node);
    }, true);

    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
        window.addEventListener(type, finish, true);
    }
    window.addEventListener("blur", reset);
}

app.registerExtension({ name: "KASKI.ShakeDisconnect", setup: installShake });
