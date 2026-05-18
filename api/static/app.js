// =========================
// FIX OSZLOPOK
// =========================
const HEADERS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "prediction"
];

// =========================
// STATE
// =========================
let data = [];

// =========================
// INIT EVENTS
// =========================
document.getElementById("upload-csv").addEventListener("change", handleFile);
document.getElementById("addRowBtn").addEventListener("click", addRow);
document.getElementById("predictBtn").addEventListener("click", predict);
document.getElementById("downloadBtn").addEventListener("click", downloadCSV);

renderTable()

// =========================
// CSV LOAD
// =========================
function handleFile(event) {

    const file = event.target.files[0];
    if (!file) return;

    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        transformHeader: h => h.trim(),
        complete: res => initFromCSV(res.data)
    });
}

// =========================
// INIT DATA
// =========================
function initFromCSV(csvData) {

    data = csvData.map(row => createEmptyRow(row));

    renderTable();
}

// =========================
// ROW FACTORY
// =========================
function createEmptyRow(row = {}) {

    const obj = {};

    HEADERS.forEach(key => {
        obj[key] = row[key] ?? "";
    });

    return obj;
}

// =========================
// ADD ROW
// =========================
function addRow() {

    data.push(createEmptyRow());

    renderTable();
}

// =========================
// UPDATE CELL
// =========================
function updateCell(input) {

    const { row, key } = input.dataset;

    data[row][key] = input.value;
}

// =========================
// DELETE ROW
// =========================
function deleteRow(index) {

    data.splice(index, 1);

    renderTable();
}

// =========================
// RENDER TABLE
// =========================
function renderTable() {

    const table = document.getElementById("data");

    table.innerHTML = "";

    renderHeader(table);

    renderRows(table);
}

function renderHeader(table) {

    let tr = "<tr>";

    HEADERS.forEach(h => {
        tr += `<th>${h}</th>`;
    });

    tr += "<th>actions</th></tr>";

    table.innerHTML += tr;
}

function renderRows(table) {

    data.forEach((row, i) => {

        let tr = "<tr>";

        HEADERS.forEach(key => {

            if (key === "prediction") {

                tr += `<td class="prediction-cell">
                    ${formatPrediction(row[key])}
                </td>`;

            } else {

                tr += `
                    <td>
                        <input
                            value="${row[key] ?? ""}"
                            data-row="${i}"
                            data-key="${key}"
                            oninput="updateCell(this)"
                        />
                    </td>
                `;
            }
        });

        tr += `
            <td>
                <button class="deleteBtn" onclick="deleteRow(${i})">
                    Törlés
                </button>
            </td>
        `;

        tr += "</tr>";

        table.innerHTML += tr;
    });
}

// =========================
// PREDICTION FORMAT
// =========================
function formatPrediction(value) {

    if (value === "" || value === undefined || value === null) return "";

    return `<span class="badge ${value == 1 ? "badge-good" : "badge-bad"}">
        ${value}
    </span>`;
}

// =========================
// PREDICT API
// =========================
async function predict() {

    try {

        const selectedModel = document.getElementById("modelSelect").value;

        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: selectedModel,
                data: data
            })
        });

        const result = await res.json();

        if (result.error) {
            alert(result.error);
            return;
        }

        data = data.map((row, i) => ({
            ...row,
            prediction: result.predictions[i]
        }));

        renderTable();

    } catch (err) {
        console.error(err);
        alert("Hiba a predikció során");
    }
}

// =========================
// CSV EXPORT
// =========================
function downloadCSV() {

    if (!data.length) {
        alert("Nincs mit letölteni!");
        return;
    }

    const csv = Papa.unparse(data, {
        columns: HEADERS
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.href = url;
    a.download = "predictions.csv";

    document.body.appendChild(a);
    a.click();

    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}