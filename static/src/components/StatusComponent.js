import { socket, get_state } from '../connections.js'

const template = `
<div>
    <form>
        <label for="status.id">ID: {{ status.id }}</label>
        <input type="checkbox" id="idle" v-model="status.idle" @change="notify">
        <label for="status.name">Name:</label>
        <input type="text" id="name" v-model="status.name" @change="notify">
        <div v-if="hasblah">
            <label for="status.blah">Blah:</label>
            <input type="text" id="blah"v-model="status.blah" @change="notify">
        </div>
    </form>
    <status-component v-if="!(!status.status)" v-bind="status" @change="notify" />
</div>
`;

export default {
    name: "status-component",
    props: ['status'],
    template,
    methods: {
        notify() {
            console.log(this.status)
            this.$emit('inputs_changed', this.status);
        }
    },
    computed: {
        hasblah() {
            return ('blah' in this.status)
        }
    },
    emits: ['inputs_changed']
};