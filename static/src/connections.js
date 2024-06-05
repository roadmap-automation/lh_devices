import { io } from 'socketio';

let socket = io()

export async function get_state(id) {
    let id_prefix = id
    if (id.length > 0) {
        id_prefix = '/' + id
    }
    let retval = await(await fetch(id_prefix + '/state')).json()
    return retval
}

export { socket }
