// dashboard.js

let globalCsvData, globalAccuracyData, globalImportanceData;
let currentTargetColumn = "";
let targetClasses = [];
let currentTaskType = "classification"; // "classification" | "regression"

const colorScale = d3.scaleOrdinal()
    .domain(["Decision Tree (Simple)", "Decision Tree (Complex)", "Random Forest", "Logistic Regression"])
    .range(["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]);

const models = ["Decision Tree (Simple)", "Decision Tree (Complex)", "Random Forest", "Logistic Regression"];

// Step 1: File Selection -> Fetch Columns
document.getElementById('csvFile').addEventListener('change', async function() {
    const file = this.files[0];
    if (!file) return;

    const status = document.getElementById('uploadStatus');
    status.textContent = '📋 Reading columns...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/columns', { method: 'POST', body: formData });
        const data = await response.json();

        const select = document.getElementById('targetColSelect');
        select.innerHTML = data.columns.map(col => `<option value="${col}">${col}</option>`).join('');

        status.textContent = `✅ Ready! Select the target column and click Train.`;
    } catch (err) {
        status.textContent = '❌ Could not read file. Is the Python server running?';
    }
});

// Step 2: Train Models -> Render Dashboard
async function trainModels() {
    const file = document.getElementById('csvFile').files[0];
    const targetCol = document.getElementById('targetColSelect').value;
    const btn = document.getElementById('trainBtn');
    const status = document.getElementById('uploadStatus');

    if (!file || !targetCol) return;

    btn.disabled = true;
    btn.textContent = '⏳ Training...';
    status.textContent = '🔄 Training four models — this takes 5-15 seconds...';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_col', targetCol);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();

        if (!data.success) {
            status.textContent = '❌ Error: ' + data.error;
            btn.disabled = false;
            btn.textContent = '🚀 Train Models';
            return;
        }

        // Bridge: Save backend data to JS memory
        globalCsvData = data.predictions;
        globalAccuracyData = data.accuracy;
        globalImportanceData = data.importance;
        currentTargetColumn = data.target_column;
        targetClasses = data.classes;
        currentTaskType = data.task_type || "classification";

        // Update UI
        document.getElementById('stat-dataset').textContent = file.name;
        document.getElementById('stat-target').textContent = currentTargetColumn;
        document.getElementById('stat-train').textContent = data.n_train;
        document.getElementById('stat-test').textContent = data.n_test;

        document.getElementById('groupSelect').innerHTML = data.groupable_columns.map(col => `<option value="${col}">${col}</option>`).join('');

        document.getElementById('statsBar').style.display = 'flex';
        document.getElementById('controlsBar').style.display = 'flex';
        document.getElementById('dashboardContent').style.display = 'grid';

        // Swap card titles/subtitles depending on task type
        updateCardLabels();

        // Set up Listeners
        d3.select("#modelSelect").on("change", updateDashboard);
        d3.select("#featureSelect").on("change", updateDashboard);
        d3.select("#groupSelect").on("change", updateDashboard);

        // Draw!
        updateDashboard();
        status.textContent = `✅ Complete! Analyzed ${data.n_test} test samples. (${currentTaskType})`;

    } catch (err) {
        status.textContent = '❌ Network error: ' + err.message;
    }
    btn.disabled = false;
    btn.textContent = '🚀 Train Models';
}

// Updates card headings so the dashboard reads correctly for either task type.
function updateCardLabels() {
    const isRegression = currentTaskType === "regression";

    document.getElementById('accuracy-card-title').textContent =
        isRegression ? "Model Error Comparison" : "Model Accuracy Comparison";
    document.getElementById('accuracy-card-subtitle').textContent =
        isRegression ? "Mean Absolute Error and R² on the held-out test set" : "Overall accuracy on the held-out test set";

    document.getElementById('panel3-card-title').textContent =
        isRegression ? "Predicted vs Actual" : "Confusion Matrices";
    document.getElementById('panel3-card-subtitle').textContent =
        isRegression ? "How close are predictions to the true values?" : "True Positives, False Positives, and Errors";

    document.getElementById('panel4-card-title').textContent =
        isRegression ? "Residuals" : "Prediction Agreement";
    document.getElementById('panel4-card-subtitle').textContent =
        isRegression ? "Error (actual − predicted) per test sample" : "How often do all 4 models guess the exact same thing?";
}

// Step 3: D3 Master Controller
function updateDashboard() {
    const selectedModel = d3.select("#modelSelect").property("value");
    const topNFeatures = parseInt(d3.select("#featureSelect").property("value"));
    const groupBy = d3.select("#groupSelect").property("value");

    if (currentTaskType === "regression") {
        drawErrorChart(globalAccuracyData, selectedModel);
        drawFeatureImportance(globalImportanceData, selectedModel, topNFeatures);
        drawGroupError(globalCsvData, groupBy, selectedModel);
        drawScatterChart(globalCsvData, selectedModel);
        drawResidualsChart(globalCsvData, selectedModel);
    } else {
        drawAccuracyChart(globalAccuracyData, selectedModel);
        drawFeatureImportance(globalImportanceData, selectedModel, topNFeatures);
        drawGroupAccuracy(globalCsvData, groupBy, selectedModel);
        drawAgreementChart(globalCsvData);
        drawConfusionMatrices(globalCsvData, selectedModel);
    }
}

// ===========================================================
// CLASSIFICATION CHARTS
// ===========================================================

function drawAccuracyChart(data, highlightedModel) {
    const svg = d3.select("#accuracy-chart");
    svg.selectAll("*").remove();

    const width = 600, height = 300;
    const margin = {top: 30, right: 20, bottom: 100, left: 70};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "100%");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().domain(data.map(d => d.model)).range([0, innerWidth]).padding(0.3);
    const minAcc = d3.min(data, d => d.accuracy) - 0.1;
    const y = d3.scaleLinear().domain([minAcc > 0 ? minAcc : 0, 1]).range([innerHeight, 0]);

    g.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x))
        .selectAll("text").attr("transform", "rotate(-45)").style("text-anchor", "end").attr("dx", "-.8em").attr("dy", ".15em").style("font-size", "14px");
    g.append("g").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));

    g.selectAll(".bar").data(data).enter().append("rect")
        .attr("fill", d => colorScale(d.model))
        .style("opacity", d => (highlightedModel === "all" || highlightedModel === d.model) ? 1 : 0.2)
        .attr("x", d => x(d.model)).attr("y", d => y(d.accuracy))
        .attr("width", x.bandwidth()).attr("height", d => innerHeight - y(d.accuracy)).attr("rx", 4);

    g.selectAll(".label").data(data).enter().append("text")
        .attr("x", d => x(d.model) + x.bandwidth() / 2).attr("y", d => y(d.accuracy) - 10)
        .attr("text-anchor", "middle").text(d => (d.accuracy * 100).toFixed(1) + "%")
        .style("font-size", "14px").style("font-weight", "bold");
}

function drawGroupAccuracy(data, groupBy, highlightedModel) {
    const grouped = d3.groups(data, d => d[groupBy]);

    const chartData = grouped.map(([groupName, records]) => {
        let result = { group: groupName, count: records.length };
        models.forEach(model => {
            let correct = records.filter(r => r[currentTargetColumn] === r[model]).length;
            result[model] = correct / records.length;
        });
        return result;
    }).filter(d => d.count > 5);

    drawGroupedBarChart("#group-chart", chartData, highlightedModel, d3.format(".0%"), [0, 1]);
}

function drawAgreementChart(data) {
    const svg = d3.select("#agreement-chart");
    svg.selectAll("*").remove();

    let allAgree = 0, threeAgree = 0, split = 0;

    data.forEach(row => {
        let counts = {};
        models.forEach(m => { counts[row[m]] = (counts[row[m]] || 0) + 1; });
        let maxAgree = Math.max(...Object.values(counts));
        if(maxAgree === 4) allAgree++;
        else if (maxAgree === 3) threeAgree++;
        else split++;
    });

    const pieData = [
        {label: "All 4 Agree", count: allAgree, color: "#4e79a7"},
        {label: "3 Agree", count: threeAgree, color: "#f28e2b"},
        {label: "Split", count: split, color: "#e15759"}
    ];

    const width = 400, height = 300, radius = Math.min(width, height) / 2 - 20;
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "100%");
    const g = svg.append("g").attr("transform", `translate(${width/2},${height/2})`);

    const pie = d3.pie().value(d => d.count);
    const arc = d3.arc().innerRadius(radius * 0.5).outerRadius(radius);

    g.selectAll("path").data(pie(pieData)).enter().append("path")
        .attr("d", arc).attr("fill", d => d.data.color).attr("stroke", "white").style("stroke-width", "2px");

    g.selectAll("text").data(pie(pieData)).enter().append("text")
        .attr("transform", d => `translate(${arc.centroid(d)})`)
        .attr("text-anchor", "middle").style("font-size", "14px").style("fill", "#fff")
        .text(d => Math.round((d.data.count / data.length) * 100) + "%");
}

function drawConfusionMatrices(data, highlightedModel) {
    const wrapper = d3.select("#confusion-wrapper");
    wrapper.selectAll("*").remove();
    wrapper.style("display", "grid");

    const modelsToDraw = highlightedModel === "all" ? models.slice(0,2) : [highlightedModel];
    const classA = targetClasses[0];
    const classB = targetClasses[1] || "Other";

    modelsToDraw.forEach(model => {
        let tp = 0, fp = 0, tn = 0, fn = 0;

        data.forEach(row => {
            let actual = row[currentTargetColumn];
            let predicted = row[model];

            if (actual === classA && predicted === classA) tp++;
            if (actual === classB && predicted === classA) fp++;
            if (actual === classB && predicted === classB) tn++;
            if (actual === classA && predicted === classB) fn++;
        });

        const matrixHTML = `
            <div style="border-top: 4px solid ${colorScale(model)}; padding: 10px; background:#f9f9f9; border-radius:5px;">
                <h4 style="margin-bottom:10px; font-size: 14px;">${model}</h4>
                <table style="width:100%; border-collapse: collapse; text-align:center; font-size:12px;">
                    <tr>
                        <td></td>
                        <td style="font-weight:bold; color:#59a14f">Pred:<br>${classA}</td>
                        <td style="font-weight:bold; color:#e15759">Pred:<br>${classB}</td>
                    </tr>
                    <tr>
                        <td style="font-weight:bold;">Actual:<br>${classA}</td>
                        <td style="background:#e8f4e8; border:1px solid #ddd; padding:5px;"><b>${tp}</b></td>
                        <td style="background:#fce8e8; border:1px solid #ddd; padding:5px;"><b>${fn}</b></td>
                    </tr>
                    <tr>
                        <td style="font-weight:bold;">Actual:<br>${classB}</td>
                        <td style="background:#fce8e8; border:1px solid #ddd; padding:5px;"><b>${fp}</b></td>
                        <td style="background:#e8f4e8; border:1px solid #ddd; padding:5px;"><b>${tn}</b></td>
                    </tr>
                </table>
            </div>
        `;
        wrapper.append("div").html(matrixHTML);
    });
}

// ===========================================================
// REGRESSION CHARTS
// ===========================================================

function drawErrorChart(data, highlightedModel) {
    const svg = d3.select("#accuracy-chart");
    svg.selectAll("*").remove();

    const width = 600, height = 300;
    const margin = {top: 30, right: 20, bottom: 100, left: 70};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "100%");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().domain(data.map(d => d.model)).range([0, innerWidth]).padding(0.3);
    const y = d3.scaleLinear().domain([0, d3.max(data, d => d.mae) * 1.25]).range([innerHeight, 0]);

    g.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x))
        .selectAll("text").attr("transform", "rotate(-45)").style("text-anchor", "end").attr("dx", "-.8em").attr("dy", ".15em").style("font-size", "14px");
    g.append("g").call(d3.axisLeft(y).ticks(5));

    g.append("text").attr("x", -innerHeight/2).attr("y", -50).attr("transform", "rotate(-90)")
        .attr("text-anchor", "middle").style("font-size", "12px").style("fill", "#666").text("Mean Absolute Error (lower is better)");

    g.selectAll(".bar").data(data).enter().append("rect")
        .attr("fill", d => colorScale(d.model))
        .style("opacity", d => (highlightedModel === "all" || highlightedModel === d.model) ? 1 : 0.2)
        .attr("x", d => x(d.model)).attr("y", d => y(d.mae))
        .attr("width", x.bandwidth()).attr("height", d => innerHeight - y(d.mae)).attr("rx", 4);

    g.selectAll(".label").data(data).enter().append("text")
        .attr("x", d => x(d.model) + x.bandwidth() / 2).attr("y", d => y(d.mae) - 10)
        .attr("text-anchor", "middle")
        .text(d => `MAE ${d.mae.toFixed(2)} | R² ${d.r2.toFixed(2)}`)
        .style("font-size", "11px").style("font-weight", "bold");
}

function drawGroupError(data, groupBy, highlightedModel) {
    const grouped = d3.groups(data, d => d[groupBy]);

    const chartData = grouped.map(([groupName, records]) => {
        let result = { group: groupName, count: records.length };
        models.forEach(model => {
            const errors = records.map(r => Math.abs(r[currentTargetColumn] - r[model]));
            result[model] = d3.mean(errors);
        });
        return result;
    }).filter(d => d.count > 5);

    const maxVal = d3.max(chartData, d => d3.max(models, m => d[m]));
    drawGroupedBarChart("#group-chart", chartData, highlightedModel, d3.format(".2f"), [0, maxVal * 1.1]);
}

function drawGroupedBarChart(selector, chartData, highlightedModel, tickFormat, yDomain) {
    const svg = d3.select(selector);
    svg.selectAll("*").remove();

    const width = 1000, height = 350;
    const margin = {top: 30, right: 20, bottom: 80, left: 60};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "100%");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x0 = d3.scaleBand().domain(chartData.map(d => d.group)).range([0, innerWidth]).padding(0.2);
    const modelsToDraw = highlightedModel === "all" ? models : [highlightedModel];
    const x1 = d3.scaleBand().domain(modelsToDraw).range([0, x0.bandwidth()]).padding(0.05);
    const y = d3.scaleLinear().domain(yDomain).range([innerHeight, 0]);

    g.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x0))
        .selectAll("text").attr("transform", "rotate(-25)").style("text-anchor", "end").style("font-size", "14px");
    g.append("g").call(d3.axisLeft(y).tickFormat(tickFormat));

    const groupSelection = g.selectAll(".group").data(chartData).enter().append("g").attr("transform", d => `translate(${x0(d.group)},0)`);

    groupSelection.selectAll("rect").data(d => modelsToDraw.map(key => ({key: key, value: d[key]}))).enter().append("rect")
        .attr("x", d => x1(d.key)).attr("y", d => y(d.value))
        .attr("width", x1.bandwidth()).attr("height", d => innerHeight - y(d.value))
        .attr("fill", d => colorScale(d.key)).attr("rx", 2);
}

function drawScatterChart(data, highlightedModel) {
    const wrapper = d3.select("#confusion-wrapper");
    wrapper.selectAll("*").remove();
    wrapper.style("display", "block");

    const svgEl = wrapper.append("svg").attr("id", "scatter-chart-svg");

    const width = 500, height = 400;
    const margin = {top: 20, right: 20, bottom: 50, left: 60};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svgEl.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "100%");
    const g = svgEl.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const modelsToDraw = highlightedModel === "all" ? models : [highlightedModel];

    const allActual = data.map(d => d[currentTargetColumn]);
    const allPred = data.flatMap(d => modelsToDraw.map(m => d[m]));
    const domainMin = Math.min(d3.min(allActual), d3.min(allPred));
    const domainMax = Math.max(d3.max(allActual), d3.max(allPred));

    const x = d3.scaleLinear().domain([domainMin, domainMax]).range([0, innerWidth]).nice();
    const y = d3.scaleLinear().domain([domainMin, domainMax]).range([innerHeight, 0]).nice();

    g.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x));
    g.append("g").call(d3.axisLeft(y));

    g.append("text").attr("x", innerWidth/2).attr("y", innerHeight + 40)
        .attr("text-anchor", "middle").style("font-size", "12px").style("fill", "#666")
        .text(`Actual ${currentTargetColumn}`);
    g.append("text").attr("x", -innerHeight/2).attr("y", -45).attr("transform", "rotate(-90)")
        .attr("text-anchor", "middle").style("font-size", "12px").style("fill", "#666")
        .text(`Predicted ${currentTargetColumn}`);

    g.append("line")
        .attr("x1", x(domainMin)).attr("y1", y(domainMin))
        .attr("x2", x(domainMax)).attr("y2", y(domainMax))
        .attr("stroke", "#999").attr("stroke-dasharray", "4,4");

    modelsToDraw.forEach(model => {
        // Fix for safe CSS class names (removes parentheses and spaces)
        const safeClassName = model.replace(/[^a-zA-Z0-9]/g, '');

        g.selectAll(`.dot-${safeClassName}`)
            .data(data).enter().append("circle")
            .attr("cx", d => x(d[currentTargetColumn]))
            .attr("cy", d => y(d[model]))
            .attr("r", 4)
            .attr("fill", colorScale(model))
            .attr("opacity", 0.6);
    });
}

function drawResidualsChart(data, highlightedModel) {
    const svg = d3.select("#agreement-chart");
    svg.selectAll("*").remove();

    const width = 500, height = 300;
    const margin = {top: 20, right: 20, bottom: 50, left: 60};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "100%");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const modelsToDraw = highlightedModel === "all" ? models : [highlightedModel];

    const residualData = data.map((d, i) => {
        const r = {index: i, actual: d[currentTargetColumn]};
        modelsToDraw.forEach(m => { r[m] = d[currentTargetColumn] - d[m]; });
        return r;
    });

    const allResiduals = residualData.flatMap(d => modelsToDraw.map(m => d[m]));
    const maxAbs = d3.max(allResiduals.map(Math.abs));

    const x = d3.scaleLinear().domain([0, data.length]).range([0, innerWidth]);
    const y = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([innerHeight, 0]).nice();

    g.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(5));
    g.append("g").call(d3.axisLeft(y));

    g.append("line")
        .attr("x1", 0).attr("y1", y(0)).attr("x2", innerWidth).attr("y2", y(0))
        .attr("stroke", "#999").attr("stroke-dasharray", "4,4");

    g.append("text").attr("x", innerWidth/2).attr("y", innerHeight + 40)
        .attr("text-anchor", "middle").style("font-size", "12px").style("fill", "#666")
        .text("Test sample index");
    g.append("text").attr("x", -innerHeight/2).attr("y", -45).attr("transform", "rotate(-90)")
        .attr("text-anchor", "middle").style("font-size", "12px").style("fill", "#666")
        .text("Residual (actual − predicted)");

    modelsToDraw.forEach(model => {
        // Fix for safe CSS class names (removes parentheses and spaces)
        const safeClassName = model.replace(/[^a-zA-Z0-9]/g, '');

        g.selectAll(`.res-${safeClassName}`)
            .data(residualData).enter().append("circle")
            .attr("cx", d => x(d.index))
            .attr("cy", d => y(d[model]))
            .attr("r", 4)
            .attr("fill", colorScale(model))
            .attr("opacity", 0.6);
    });
}

// ===========================================================
// SHARED CHART
// ===========================================================

function drawFeatureImportance(data, highlightedModel, topN) {
    const svg = d3.select("#importance-chart");
    svg.selectAll("*").remove();

    const modelToShow = highlightedModel === "all" ? "Random Forest" : highlightedModel;
    if(modelToShow === "Logistic Regression") return;

    let filteredData = data.filter(d => d.model === modelToShow).sort((a, b) => b.importance - a.importance).slice(0, topN);

    const width = 600, height = 300;
    const margin = {top: 20, right: 30, bottom: 40, left: 200};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "100%");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, d3.max(filteredData, d => d.importance)]).range([0, innerWidth]);
    const y = d3.scaleBand().domain(filteredData.map(d => d.feature)).range([0, innerHeight]).padding(0.2);

    g.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(5));
    g.append("g").call(d3.axisLeft(y)).style("font-size", "12px");

    g.selectAll(".bar").data(filteredData).enter().append("rect")
        .attr("fill", colorScale(modelToShow))
        .attr("y", d => y(d.feature)).attr("x", 0)
        .attr("height", y.bandwidth()).attr("width", d => x(d.importance)).attr("rx", 3);
}