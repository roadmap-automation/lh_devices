import { createApp } from 'vue';

import PlotComponent from './components/PlotComponent.js';
import FormComponent from './components/FormComponent.js';

const app = createApp({
    data: () => ({
        traces: [{ x: [0,1,2], y: [3,2,4], type: 'xy' }],
        form_inputs: {
            name: "solvent",
            value: 32.0
        },
    }),
    components: {
        PlotComponent,
        FormComponent
    },
    template: `
        <div>
            <h2>Vue.js App</h2>
            <form-component @inputs_changed="onFormInputsChanged" />
            <plot-component :traces="traces" />
        </div>
    `,
    methods: {
        onFormInputsChanged(inputs) {
            alert(`Received: ${inputs.name} = ${inputs.value}`);
        }
    }
});

app.mount('#app');