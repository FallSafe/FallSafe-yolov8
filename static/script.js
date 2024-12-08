document.getElementById("saveEmail").addEventListener("click", () => {
    const email = document.getElementById("recipientEmail").value.trim();
    if (email) {
        fetch('/send_email', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: email })
        })
        .then(response => {
            if (!response.ok) throw new Error("Invalid email format.");
            return response.json();
        })
        .then(data => {
            alert(data.message);
            document.getElementById("videoStream").style.display = "block";
            document.getElementById("fallStatus").innerText = "Live feed activated.";
        })
        .catch(error => alert(`Error: ${error.message}`));
    } else {
        alert("Please enter a valid email address.");
    }
});

function updateFallStatus() {
    fetch('/fall_status')
        .then(response => response.json())
        .then(data => {
            document.getElementById("fallStatus").innerText = data.status;
            document.getElementById("alertMessage").style.display = 
                data.status.includes("ALERT") ? "block" : "none";
        })
        .catch(error => console.error('Error updating fall status:', error));
}

setInterval(updateFallStatus, 1000);
