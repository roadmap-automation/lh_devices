const template = `
<div>
    <form>
        <label for="name">Name:</label>
        <input type="text" id="name" v-model="form_inputs.name" @change="notify">
        <label for="value">Value:</label>
        <input type="number" id="value" v-model="form_inputs.value" @change="notify">
    </form>
</div>
`;

export default {
    name: "form-component",
    data: () => ({
        form_inputs: {
            name: "solvent",
            value: 12.0
        }
    }),
    template,
    methods: {
        notify() {
            this.$emit('inputs_changed', this.form_inputs);
        }
    },
    emits: ['inputs_changed']
};