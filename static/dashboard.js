document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const captureBtn = document.getElementById('captureBtn');
    const instructionLabel = document.getElementById('instructionLabel');
    const resultLabel = document.getElementById('resultLabel');
    const historyTableBody = document.getElementById('historyTableBody');
    const registrationForm = document.getElementById('registrationForm');
    const personNameInput = document.getElementById('personName');
    const registerBtn = document.getElementById('registerBtn');

    let livenessCheckInterval;
    let lastCapturedPhoto = null; // Store the photo from the last capture

    startBtn.addEventListener('click', async () => {
        // Start the liveness check on the server
        await fetch('/start_liveness_check', { method: 'POST' });
        
        // Hide start button and show capture button
        startBtn.classList.add('hidden');
        captureBtn.classList.remove('hidden');

        // Hide registration form
        registrationForm.classList.add('hidden');
        personNameInput.value = '';

        // Start polling the server for liveness status
        livenessCheckInterval = setInterval(async () => {
            const response = await fetch('/get_liveness_status');
            const status = await response.json();
            
            updateLivenessStatus(status);
            
            if (status.status !== 'pending') {
                clearInterval(livenessCheckInterval);
                if (status.status === 'live') {
                    instructionLabel.textContent = 'Liveness check passed! Click "Capture Photo" to proceed.';
                    instructionLabel.classList.remove('text-blue-600');
                    instructionLabel.classList.add('text-green-600');
                } else {
                    instructionLabel.textContent = status.label;
                    instructionLabel.classList.remove('text-blue-600');
                    instructionLabel.classList.add('text-red-600');
                    // Show start button again to allow a retry
                    startBtn.classList.remove('hidden');
                    captureBtn.classList.add('hidden');
                }
            }
        }, 500); // Poll every 500ms
    });

    captureBtn.addEventListener('click', async () => {
        instructionLabel.textContent = "Processing...";
        resultLabel.textContent = "-";
        resultLabel.classList.add('text-gray-600');
        
        try {
            const response = await fetch('/capture_and_process', {
                method: 'POST'
            });
            const result = await response.json();
            
            if (response.ok) {
                updateResult(result);
                addHistoryEntry(result);

                if (result.status === 'unknown') {
                    // Show registration form for unique face
                    registrationForm.classList.remove('hidden');
                    instructionLabel.textContent = "Please enter a name for this new person.";
                    lastCapturedPhoto = result.photo; // Store photo data
                    captureBtn.classList.add('hidden');
                } else {
                    // Reset UI for the next session
                    resetUI();
                }

            } else {
                resultLabel.textContent = `Error: ${result.message || 'Server error'}`;
                resultLabel.classList.remove('text-green-600');
                resultLabel.classList.add('text-red-600');
                resetUI();
            }
        } catch (error) {
            console.error('Network or server error:', error);
            resultLabel.textContent = 'Network Error';
            resultLabel.classList.remove('text-green-600');
            resultLabel.classList.add('text-red-600');
            resetUI();
        }
    });
    
    

registerBtn.addEventListener('click', async () => {
    const name = personNameInput.value.trim();
    if (name === "" || !lastCapturedPhoto) {
        alert("Please enter a name.");
        return;
    }

    try {
        const response = await fetch('/register_face', {
            method: 'POST',
            // Add this crucial header!
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                photo: lastCapturedPhoto
            })
        });
        const result = await response.json();

        if (response.ok) {
            alert(result.message);
            // Add the new entry to the history table immediately
            addHistoryEntry({
                status: "known", 
                result_label: `REGISTERED: ${name}`, 
                photo: result.photo || lastCapturedPhoto
            });
            resetUI();
        } else {
            alert(`Registration failed: ${result.message}`);
        }

    } catch (error) {
        console.error('Registration error:', error);
        alert("Failed to register. Network error.");
        resetUI();
    }
});

// ... (all your other functions)

    function resetUI() {
        instructionLabel.textContent = "Click 'Start Liveness Check' to begin.";
        instructionLabel.classList.remove('text-red-600', 'text-green-600');
        instructionLabel.classList.add('text-blue-600');
        startBtn.classList.remove('hidden');
        captureBtn.classList.add('hidden');
        registrationForm.classList.add('hidden');
        personNameInput.value = '';
    }

    function updateLivenessStatus(status) {
        if (status.status === 'pending') {
            const timeFormatted = status.time_left.toFixed(1);
            instructionLabel.textContent = `Liveness Check: ${status.current_action} (${timeFormatted}s left)`;
            instructionLabel.classList.remove('text-red-600', 'text-green-600');
            instructionLabel.classList.add('text-blue-600');
        }
        resultLabel.textContent = status.label;
    }

    function updateResult(result) {
        let labelText = '';
        let statusClass = 'text-gray-600';
        if (result.status === 'known') {
            labelText = result.result_label;
            statusClass = 'text-green-600';
        } else if (result.status === 'unknown') {
            labelText = 'UNIQUE FACE';
            statusClass = 'text-green-600';
        } else {
            labelText = result.result_label;
            statusClass = 'text-red-600';
        }
        resultLabel.textContent = labelText;
        resultLabel.className = `ml-1 ${statusClass}`;
    }
    
    function addHistoryEntry(result) {
        const now = new Date();
        const formattedTime = now.toLocaleDateString() + ' ' + now.toLocaleTimeString();

        const newRow = document.createElement('tr');
        const statusColor = result.status === 'known' || result.status === 'unknown' ? 'text-green-600' : 'text-red-600';
        const photoUrl = result.photo || 'https://storage.googleapis.com/workspace-0f70711f-8b4e-4d94-86f1-2a93ccde5887/image/121fd048-cb55-4e1b-878b-fa846adb8f41.png';
        
        newRow.innerHTML = `
            <td class="py-2 px-3">
                <img src="${photoUrl}" alt="Face" class="w-12 h-12 rounded-full object-cover">
            </td>
            <td class="py-2 px-3 text-gray-500">${formattedTime}</td>
            <td class="py-2 px-3 text-gray-500">Live Camera</td>
            <td class="py-2 px-3 font-semibold status-cell ${statusColor}">${result.result_label}</td>
        `;

        if (historyTableBody.firstChild) {
            historyTableBody.insertBefore(newRow, historyTableBody.firstChild);
        } else {
            historyTableBody.appendChild(newRow);
        }
    }
});