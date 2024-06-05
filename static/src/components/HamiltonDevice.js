import Valve from './Valve.js'
import Syringe from './Syringe.js'
import { socket } from '../connections.js';

const template = `
<div class="card">
    <div class="card-header">
        <h4>{{ name }}</h4>
        <h5>{{ id }}</h5>
    </div>
    <div v-if="!!state" class="row card-body">
        <valve class="col" v-bind="{valve: state.valve}" @changed="onValveChanged" />
        <syringe v-if="(syringe in state)" class="col" v-bind="{syringe: state.syringe, idle: state.idle}" />
    </div>
 </div>
`;

const cruft = `
    <valve v-if="!!state" v-bind="{valve: valve}" @changed="onValveChanged" />
`

export default {
    name: "hamilton-device",
    props: ['state', 'controls', 'id', 'config', 'name'],
    template,
    components: {
        Valve,
        Syringe
    },
    methods: {
        notify() {
            this.$emit('changed', this.valve.valve_position);
        },
        clicked() {
            this.$emit('clicked', this.valve.valve_position);
        },
        onValveChanged(position) {
            socket.emit(this.id, {"command": "move_valve", "data": {"index": position}})
        }

    }
};