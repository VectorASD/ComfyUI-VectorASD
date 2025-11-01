// Copyright 2025 VectorASD
// Licensed under the Apache License, Version 2.0 (see LICENSE file in project root).

import { app } from "../../../scripts/app.js";
//import { GET_CONFIG } from '../../../services/litegraphService.js' (not work)



function resort(obj, names) {
    const classes = Array(names.length);
    for (let i = 0; i < names.length; i++) classes[i] = [];
    const others = [];

    for (const item of obj) {
        const name = item.name;
        let pos = 0, other = true;
        for (const class_name of names) {
            if (name.startsWith(class_name)) {
                classes[pos].push(item);
                other = false;
                break;
            }
            pos++;
        }
        if (other) others.push(item);
    }

    const result = [];
    for (const items of classes) result.push(...items);
    result.push(...others);
    return result;
}

function counter(obj, name) {
    let count = 0;
    for (const item of obj)
        if (item.name.startsWith(name)) count++;
    return count;
}

function limiter(obj, name, limit) {
    let count = 0;
    const items = [], others = [];

    for (const item of obj)
        if (item.name.startsWith(name)) {
            if (count < limit) items.push(item);
            count++;
        } else others.push(item);

    items.push(...others);
    return items;
}



app.vectorasd = []

app.registerExtension({ 
	name: "VectorASD.jsnodes",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		app.vectorasd.push([nodeType, nodeData]);

		if (!nodeData?.category?.startsWith("VectorASD")) return;
		// console.log("VectorASD.beforeRegisterNodeDef:", nodeType, "|", nodeData, "|", app)

		switch (nodeType.comfyClass) {
			case "ASD_JoinStringMulti":
				// console.log("class:", nodeType);
				// console.log("data:", nodeData);
				nodeType.prototype.onNodeCreated = function () {
					// console.log("this:", this);
					this.addWidget("button", "Update inputs", null, () => {
						const count = counter(this.inputs, "any");
						const target = this.widgets.find(w => w.name === "inputcount")["value"];

						if (target < count)
							this.inputs = limiter(this.inputs, "any", target);
						else {
							for (let i = count + 1; i <= target; i++)
								this.addInput(`any${i}`, "*", {shape: 7});
							this.inputs = resort(this.inputs, ["any"]);
						}
					});
					let button = this.widgets.pop();
					this.widgets.unshift(button)
				}
				break;
			case "ASD_JoinStringMultiPrefix":
				nodeType.prototype.onNodeCreated = function () {
					this.constructor.prototype.addInputWidget = function (spec) {
						const proto = Object.getPrototypeOf(this);
						const method = Object.getOwnPropertyDescriptors(proto)["#addInputWidget"]?.value;
						if (method) method.call(this, spec);
					};
					console.log("node:", this);

					this.addWidget("button", "Update inputs", null, () => {
						const count = counter(this.inputs, "any");
						const target = this.widgets.find(w => w.name === "inputcount")["value"];

						if (target < count) {
							this.inputs = limiter(this.inputs, "any", target);
							this.inputs = limiter(this.inputs, "prefix", target);
							this.widgets = limiter(this.widgets, "prefix", target);
						} else {
							for (let i = count + 1; i <= target; i++) {
								this.addInput(`any${i}`, "*", {shape: 7}); // так и не смог заставить это нормально работать! :///
								// const widget = { name: input.name, [GET_CONFIG]: () => [input.type, {}] }
								// const widget = this.addWidget("STRING", `prefix${i}`, `any${i}`);
								// this.addInput(`prefix${i}`, "STRING", {shape: 7});//, widget: widget});
								// const GET_CONFIG = Object.getOwnPropertySymbols(this.inputs.filter(x => x.name == "prefix1")[0].widget)[0];
								// debugger;
								this.addInputWidget({type: "STRING", name: `prefix${i}`, isOptional: true, default: `any${i}`, multiline: false});
								//this.addInput(`prefix${i}`, "STRING", {shape: 7, widget: {
							    //		name: `prefix${i}`,
							    //		[GET_CONFIG]: () => ({type: "STRING", name: `prefix${i}`, isOptional: true, default: `any${i}`, multiline: false})
								//}});
								this.addWidget("STRING", `prefix${i}`, `any${i}`);
							}
							this.inputs = resort(this.inputs, ["any", "button", "inputcount", "prefix"]);
						}
						console.log([this.inputs, this.widgets]);
					});
					let button = this.widgets.pop();
					this.widgets.unshift(button);
					this.inputs = resort(this.inputs, ["any", "button", "inputcount", "prefix"]);
				}
				break;
		}
	},

	async setup() { // после всех beforeRegisterNodeDef
		console.log("VectorASD.jsnodes: Setup complete!")
	}
})
