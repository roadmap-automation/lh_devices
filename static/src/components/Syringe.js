const template = `
<div>
    <h5>Syringe</h5>
    <div id="syringe_progress" class="row">
        <div class="col">
            <div class="progress" role="progressbar" style="height: 100%">
                <div class="progress-bar progress-bar-striped bg-success overflow-visible text-dark" :class="[{'progress-bar-animated': !idle}]" :style="pctTravel"> {{ formatVolume }} </div>
            </div>
        </div>
        <div class="col-auto text-wrap">{{ formatTotalVolume }}</div>    
    </div>
</div>
`;

function format_volume(volume) {
    // volume in mL
    if (volume < 1.0) {
        return (volume * 1000).toFixed(1) + ' ' + String.fromCharCode(181) + 'L'
    }
    else {
        return (volume * 1).toFixed(3) + ' mL'
    }
}

export default {
    name: "syringe",
    props: ['syringe', 'idle'],
    template,
    computed: {
        pctTravel() {
            return {width: ((parseFloat(this.syringe.position) + 1.0) / parseFloat(this.syringe.syringe_volume) * 100) + '%'}
        },
        formatVolume() {
            return format_volume(parseFloat(this.syringe.position))
        },
        formatTotalVolume() {
            return format_volume(parseFloat(this.syringe.syringe_volume))
        },
        pathOptions(){
            return this.valve.valve_svg_paths
        }        
    }
};