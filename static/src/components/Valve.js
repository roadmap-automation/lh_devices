const template = `
<div>
    <h5>Valve</h5>
    <svg style="width: 100%" @click="clicked">
        <svg v-html="valve.valve_svg_paths[valve.valve_position]"/>
        <svg v-html="valve.valve_svg_template"/>
    </svg>
    <div class="row">
        <label class="col-auto col-form-label" for="valve_position_select">Position: </label>
        <select class="col form-select" id="valve_position_select" v-model="valve.valve_position" @change="notify" >
            <option v-for="(position, i) in pathOptions" :value="i">
                {{ i }}
            </option>
        </select>    
    </div>
</div>
`;

export default {
    name: "valve",
    props: ['valve'],
    template,
    methods: {
        notify() {
            this.$emit('changed', this.valve.valve_position);
        },
        clicked() {
            this.$emit('changed', (this.valve.valve_position + 1) % (this.valve.number_positions + 1));
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