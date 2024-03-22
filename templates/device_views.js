var socket = io()

function issue_command(command, data) {
    console.log('sending', {'command': command,
                            'data': data})
    socket.emit(id, {'command': command,
                     'data': data})
}


function update_controls(div_id, controls) {
    currentdiv = $('#' + div_id)
    currentdiv.empty()
    for (const [cid, cdata] of Object.entries(controls)) {
        //console.log(cid, cdata)
        ctype = cdata['type']
        var newctrl = null
        if (ctype == 'button') {
            var newctrl = $('<button>')
                            .attr("id", cid)
                            .text(cdata['text'])
                            .on('click', function () {
                                issue_command(cid, {})
                            })
        }
        else if (ctype=='select') {
            var newctrl = $('<div>')
                            .attr("id", cid)
                            .text(cdata['text']);
            var newselect = $('<select>')
                            .attr("id", cid + "_select")
            for (const option of cdata['options']) {
                $("<option>").text(option).appendTo(newselect)
            }
            newselect.val(cdata['current'])
            newselect.on("change", function () {
                issue_command(cid, {'index': newselect.prop('selectedIndex')})
            })
            newctrl.append(newselect)
        }
        currentdiv.append(newctrl)
    }
}

function update_modes(currentdiv, modes, current_mode) {
    modediv = currentdiv.find("#modes")
    modediv.empty().append("Select mode: ")
    var newselect = $('<select>')
                            .attr("id", "mode_select")
    $("<option>").text("").appendTo(newselect)
    for (const mode of modes) {
        $("<option>").text(mode).appendTo(newselect)
    }
    if (modes.includes(current_mode)) {
        newselect.val(current_mode)
    }
    newselect.on("change", function () {
        var val = newselect.val()
        if (val != "") {
            issue_command("change_mode", val)
        }
    })
    modediv.append(newselect)
}

function update_device(currentdiv, data) {
    currentdiv.find("#device_name").empty().append(data['name']);
    currentdiv.find("#config").empty().append(JSON.stringify(data['config']));
    currentdiv.find("#state").empty().append(JSON.stringify(data['state']));
    update_controls(currentdiv, data['controls'])

}

function update_assembly(currentdiv, data) {
    current_div.find("#assembly_name").empty().append(data['name'] + "<br>" + data['id']);
    update_modes(currentdiv, data['modes'], data['current_mode'])
}

function update_assembly_devices(currentdiv, devices) {
    devicediv = currentdiv.find("#devices")
    devicediv.empty()
    for (const [device_id, device_name] of Object.entries(devices)) {
        console.log(device_id, device_name)
        var newref = new $("<a>")
            .attr("href", device_id + "/")
            .text(device_name);
        devicediv.append(newref)
        devicediv.append("<br>")
    }
}

function update(id) {
    fetch(window.location.href + id + '/state')
        .then(function (response) {
            return response.json();
        }).then(function (data) {
            //console.log(data)
            currentdiv = $('#' + data['id'])
            if ("controls" in data) {
                currentdiv = $('#' + data['id'])
                update_device(data)
            }
            else if ("devices" in data) {
                currentdiv = $('#' + data['id'])
                update_assembly(data)
            }
            
        });
    }