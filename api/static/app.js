// =========================
// FIX OSZLOPOK
// =========================
const COLUMNS = [
    {
        "name": "age",
        "type": "number"
    },
    {
        "name": "workclass",
        "type": "choice",
        "options": [
            "Federal-gov",
            "Local-gov",
            "Never-worked",
            "Private",
            "Self-emp-inc",
            "Self-emp-not-inc",
            "State-gov",
            "Without-pay"
        ]
    },
    {
        "name": "fnlwgt",
        "type": "number"
    },
    {
        "name": "education",
        "type": "choice",
        "options": [
            "10th",
            "11th",
            "12th",
            "1st-4th",
            "5th-6th",
            "7th-8th",
            "9th",
            "Assoc-acdm",
            "Assoc-voc",
            "Bachelors",
            "Doctorate",
            "HS-grad",
            "Masters",
            "Preschool",
            "Prof-school",
            "Some-college"
        ]
    },
    {
        "name": "marital.status",
        "type": "choice",
        "options": [
            "Divorced",
            "Married-AF-spouse",
            "Married-civ-spouse",
            "Married-spouse-absent",
            "Never-married",
            "Separated",
            "Widowed"
        ]
    },
    {
        "name": "occupation",
        "type": "choice",
        "options": [
            "Adm-clerical",
            "Armed-Forces",
            "Craft-repair",
            "Exec-managerial",
            "Farming-fishing",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Other-service",
            "Priv-house-serv",
            "Prof-specialty",
            "Protective-serv",
            "Sales",
            "Tech-support",
            "Transport-moving"
        ]
    },
    {
        "name": "relationship",
        "type": "choice",
        "options": [
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Own-child",
            "Unmarried",
            "Wife"
        ]
    },
    {
        "name": "race",
        "type": "choice",
        "options": [
            "Amer-Indian-Eskimo",
            "Asian-Pac-Islander",
            "Black",
            "Other",
            "White"
        ]
    },
    {
        "name": "sex",
        "type": "choice",
        "options": [
            "Female",
            "Male"
        ]
    },
    {
        "name": "capital.gain",
        "type": "number"
    },
    {
        "name": "capital.loss",
        "type": "number"
    },
    {
        "name": "hours.per.week",
        "type": "number"
    },
    {
        "name": "native.country",
        "type": "choice",
        "options": [
            "Cambodia",
            "Canada",
            "China",
            "Columbia",
            "Cuba",
            "Dominican-Republic",
            "Ecuador",
            "El-Salvador",
            "England",
            "France",
            "Germany",
            "Greece",
            "Guatemala",
            "Haiti",
            "Holand-Netherlands",
            "Honduras",
            "Hong",
            "Hungary",
            "India",
            "Iran",
            "Ireland",
            "Italy",
            "Jamaica",
            "Japan",
            "Laos",
            "Mexico",
            "Nicaragua",
            "Outlying-US(Guam-USVI-etc)",
            "Peru",
            "Philippines",
            "Poland",
            "Portugal",
            "Puerto-Rico",
            "Scotland",
            "South",
            "Taiwan",
            "Thailand",
            "Trinadad&Tobago",
            "United-States",
            "Vietnam",
            "Yugoslavia"
        ]
    },
    {
        "name": "prediction",
        "type": "result"
    }
]

const EDUCATION_MAP = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
};

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
        complete: res => {
            initFromCSV(res.data)
            event.target.value = "";
        }
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

    COLUMNS.forEach(column => {
        if (!row[column.name] || row[column.name] == "?") {
            obj[column.name] = ""
        } else {
            obj[column.name] = row[column.name]
        }
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

    tr += "<th style='padding: 10px 20px;'>actions</th>";

    COLUMNS.forEach(column => {
        tr += `<th style='padding: 10px 20px;'>${column.name}</th>`;
    });

    tr += "</tr>";

    table.innerHTML += tr;
}

function renderRows(table) {

    data.forEach((row, i) => {

        let tr = "<tr>";

        tr += `
            <td>
                <button class="deleteBtn" onclick="deleteRow(${i})">
                    Törlés
                </button>
            </td>
        `;

        COLUMNS.forEach(column => {

            if (column.type == "result") {
                tr += `<td class="prediction-cell">
                    ${formatPrediction(row[column.name])}
                </td>`;
            } else if (column.type == "number") {
                tr += `
                    <td>
                        <input
                            type="number"
                            value="${row[column.name]}"
                            placeholder="?"
                            data-row="${i}"
                            data-key="${column.name}"
                            oninput="updateCell(this)"
                        />
                    </td>
                `;
            } else if (column.type == "choice") {
                const optionsHtml = column.options.map(option => {
                    const isSelected = row[column.name] === option ? 'selected' : '';
                    return `<option value="${option}" ${isSelected}>${option}</option>`;
                }).join('');

                tr += `
                    <td>
                        <select
                            data-row="${i}"
                            data-key="${column.name}"
                            onchange="updateCell(this)"
                        >
                            <option value="" selected>
                                ?
                            </option>
                            ${optionsHtml}
                        </select>
                    </td>
                `; 
            }
        });

        tr += "</tr>";

        table.innerHTML += tr;
    });
}

// =========================
// PREDICTION FORMAT
// =========================
function formatPrediction(value) {

    if (value === "" || value === undefined || value === null) return "";

    return `<span class="badge ${value >= 0.5 ? "badge-good" : "badge-bad"}">
        ${value.toFixed(3)}
    </span>`;
}

// =========================
// PREDICT API
// =========================
async function predict() {

    try {

        const selectedModel = document.getElementById("modelSelect").value;

        const dataForBackend = data.map(row => {
            const calculatedNum = EDUCATION_MAP[row["education"]] ?? row["education.num"];
            
            return {
                ...row,
                "education.num": calculatedNum ?? ""
            };
        });

        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: selectedModel,
                data: dataForBackend
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
        columns: COLUMNS.forEach((column) => column.name)
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