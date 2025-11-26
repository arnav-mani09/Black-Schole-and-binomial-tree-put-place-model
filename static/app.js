const $ = (id) => document.getElementById(id);

const parseRange = (input) => {
  const [min, max] = input.split(",").map((v) => parseFloat(v.trim()));
  return [min, max];
};

const collectParams = () => ({
  model: $("model").value,
  option_type: $("optionType").value,
  spot: parseFloat($("spot").value),
  strike: parseFloat($("strike").value),
  maturity: parseFloat($("maturity").value),
  rate: parseFloat($("rate").value),
  vol: parseFloat($("vol").value),
  steps: parseInt($("steps").value, 10),
});

const postJSON = async (url, payload) => {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
};

$("priceBtn").addEventListener("click", async () => {
  try {
    const params = collectParams();
    const result = await postJSON("/api/price", params);
    $("priceOutput").textContent = `Price: ${result.price.toFixed(4)}`;
  } catch (error) {
    $("priceOutput").textContent = error.message;
  }
});

const drawHeatmap = (data) => {
  Plotly.newPlot(
    "heatmap",
    [
      {
        z: data.surface,
        x: data.spots,
        y: data.vols,
        type: "heatmap",
        colorscale: "Viridis",
      },
    ],
    {
      title: "Option Price Heatmap",
      xaxis: { title: "Spot Price" },
      yaxis: { title: "Volatility" },
    },
    { displayModeBar: false }
  );
};

$("heatmapBtn").addEventListener("click", async () => {
  try {
    const params = collectParams();
    params.spot_range = parseRange($("spotRange").value);
    params.vol_range = parseRange($("volRange").value);
    params.grid = parseInt($("grid").value, 10);

    const data = await postJSON("/api/heatmap", params);
    drawHeatmap(data);
  } catch (error) {
    console.error(error);
  }
});

// Render default heatmap on load
$("heatmapBtn").click();
