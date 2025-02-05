document.getElementById("saveDetails").addEventListener("click", () => {
  const email = document.getElementById("recipientEmail").value.trim();
  const phone = document.getElementById("phoneNumber").value.trim();
  const conf = document.getElementById("confidence").value.trim();
  console.log("Saving details:", email, phone, conf);

  if (email && phone && conf) {
    fetch("/send_details", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email, phone: phone, conf: conf }),
    })
      .then((response) => {
        if (!response.ok) throw new Error("Invalid Details");
        return response.json();
      })
      .then((data) => {
        alert(data.message);
        document.getElementById("videoStream").style.display = "block";
        document.getElementById("fallStatus").innerText =
          "Live feed activated.";
        document.getElementById("fallStatus").style.display = "block";
      })
      .catch((error) => alert(`Error: ${error.message}`));
  } else {
    alert(
      "Please enter a valid Email address, Phone Number, and Confidence threshold"
    );
  }
});

function updateFallStatus() {
  const email = document.getElementById("recipientEmail").value.trim();
  const phone = document.getElementById("phoneNumber").value.trim();
  const conf = document.getElementById("confidence").value.trim();
  fetch("/fall_status")
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("fallStatus").innerText = data.status
        ? "Fall Detected"
        : "No Fall Detected";
      document.getElementById("alertMessage").style.display = data.status
        ? "block"
        : "none";
      document.getElementById("SavedEmail").style.display = data.status
        ? "block"
        : "none";
      document.getElementById("SavePhone").style.display = data.status
        ? "block"
        : "none";
      document.getElementById("SavedConf").style.display = data.status
        ? "block"
        : "none";
      document.getElementById("SavedEmail").innerText = data.status
        ? email
        : "Could not get Data";
      document.getElementById("SavePhone").innerText = data.status
        ? phone
        : "Could not get Data";
      document.getElementById("SavedConf").innerText = data.status
        ? conf
        : "Could not get Data";
      // Use class "re" to show/hide input groups.
      let reElements = document.getElementsByClassName("re");
      for (let el of reElements) {
        el.style.display = data.status ? "none" : "block";
      }
    })
    .catch((error) => console.error("Error updating fall status:", error));
}

setInterval(updateFallStatus, 500);
