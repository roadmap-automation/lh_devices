import { Valve } from './Valve.js'

const template = `
<div class="card">
    <label for="valve_position_select">Valve position:</label>
    <svg style="width: 30%" @click="clicked">
        <svg v-html="valve.valve_svg_paths[valve.valve_position]"/>
        <svg v-html="valve.valve_svg_template"/>
    </svg>
    <select id="valve_position_select" v-model="valve.valve_position" @change="notify" >
        <option v-for="(position, i) in pathOptions" :value="i">
            {{ i }}
        </option>
    </select>    
</div>
`;

export default {
    name: "device",
    props: ['device'],
    template,
    methods: {
        notify() {
            this.$emit('changed', this.valve.valve_position);
        },
        clicked() {
            this.$emit('clicked', this.valve.valve_position);
        }

    },
    computed: {
        selectedPath() {
            return this.valve.valve_svg_paths[this.valve.valve_position]
        },
        pathOptions(){
            return this.valve.valve_svg_paths
        }        
    }
};