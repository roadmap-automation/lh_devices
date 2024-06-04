import { createApp, ref } from 'vue';
import { socket, get_state } from './connections.js';

const app_id = ref('')

const status = ref({id: '',
                     idle: false,
                     name: 'alice',
                     type: 'none',
                     status: {id: '',
                     idle: false,
                     name: 'bob',
                     type: 'none',
                     status: null}})

socket.on('connect', async () => {
    status.value = await get_state('')
    app_id.value = status.value.id

    socket.on(app_id.value, async () => {
        status.value = await get_state(app_id.value)
     })
})

//import PlotComponent from './components/PlotComponent.js';
//import FormComponent from './components/FormComponent.js';
import StatusComponent from './components/StatusComponent.js';
import Valve from './components/Valve.js'

const app = createApp({
    data: () => ({
        status: {status: status},
    }),
    components: {
        StatusComponent,
        Valve
    },
    template: `
        <div>
            <h2>Vue.js App</h2>
            <status-component v-bind="status" @inputs_changed="onFormInputsChanged" />
            <valve v-if="'valve' in status.status" v-bind="{valve: status.status.valve}" @changed="onValveChanged" />
        </div>
    `,
    methods: {
        onFormInputsChanged(inputs) {
            //alert(`Received: ${inputs.name} = ${inputs.value}`);
            //console.log(inputs)
            socket.emit(app_id.value, {"command": "update_status", "data": {"status": inputs.idle}})
        },
        onValveChanged(position) {
            socket.emit(app_id.value, {"command": "move_valve", "data": {"position": position}})
        }
    }
});

app.mount('#app');