import Plotly from 'plotly';

const template = `
<div>
    <div ref="plotdiv"></div>
</div>
`;

export default {
    name: "plot-component",
    props: ['traces'],
    template,
    mounted() {
        console.log('mounted', this.traces);
        Plotly.react(this.$refs.plotdiv, this.traces);
    },
    watch: {
        traces() {
            Plotly.react(this.$refs.plotdiv, this.traces);
        }
    }
};