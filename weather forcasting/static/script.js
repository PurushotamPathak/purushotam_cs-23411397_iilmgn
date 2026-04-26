function predict() {
    let city = document.getElementById("city").value;

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city: city })
    })
    .then(res => res.json())
    .then(data => {

        if (data.error) {
            document.getElementById("result").innerText = data.error;
            return;
        }

        let output = "City: " + data.city + "\n\n";

        output += "📅 Past Data:\n";
        output += "-------------------------\n";

        for (let i = 0; i < data.dates.length; i++) {
            output += data.dates[i] + " : " + data.temps[i] + " °C\n";
        }

        output += "\n🔮 Prediction:\n";
        output += "-------------------------\n";

        for (let i = 0; i < data.future_dates.length; i++) {
            output += data.future_dates[i] + " : " + data.future_preds[i] + " °C\n";
        }

        document.getElementById("result").innerText = output;
    });
}