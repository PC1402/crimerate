document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('predictionForm');

  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    // Collect the form data and construct a JSON object
    const city = document.getElementById('city').value;
    const date = document.getElementById('date').value;
    const time = document.getElementById('time').value;

    const requestData = {
      city: city,
      date: date,
      time: time
    };

    try {
      // Send the data as a JSON payload
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',  // Ensure you're sending JSON
        },
        body: JSON.stringify(requestData)  // Convert the data into a JSON string
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      // Parse the response from JSON
      const result = await response.json();

      // Display the result in the DOM
      document.getElementById('result').innerHTML = `
        <h3>Predicted Crime Information:</h3>
        <p><strong>Crime Description:</strong> ${result["Crime Description"]}</p>
        <p><strong>Victim Age:</strong> ${result["Victim Age"]}</p>
        <p><strong>Victim Gender:</strong> ${result["Victim Gender"]}</p>
        <p><strong>Crime Domain:</strong> ${result["Crime Domain"]}</p>
      `;
    } catch (error) {
      // Show an error message if the request fails
      document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
  });
});
