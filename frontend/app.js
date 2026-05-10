let riskData = {};
let dashboardData = {};

/* =========================
   LOAD RISK DATA
========================= */

fetch("risk_data.json")
    .then(response => response.json())
    .then(data => {

        data.forEach(item => {

            riskData[item.municipio] = item;

        });

        loadMap();

    });


/* =========================
   MAP CONFIGURATION
========================= */

const map = L.map('map', {

    scrollWheelZoom: false,

    dragging: false,

    doubleClickZoom: false,

    boxZoom: false,

    keyboard: false,

    zoomControl: false,

    touchZoom: false

});


/* =========================
   LOAD MAP
========================= */

function loadMap() {

    fetch("Jalisco.json")
        .then(response => response.json())
        .then(data => {

            const geojsonLayer = L.geoJSON(data, {

                style: function(feature) {

                    const municipio =
                        feature.properties.NOMGEO;

                    const municipalityData =
                        riskData[municipio];

                    const risk =
                        municipalityData?.predicted_pd;

                    return {

                        color: "#343333",

                        weight: 1,

                        fillColor: getColor(risk),

                        fillOpacity: 0.8
                    };
                },

                onEachFeature: function(feature, layer) {

                    const municipalityCard =
                        document.getElementById("municipality-card");

                    const municipalityName =
                        document.getElementById("municipality-name");

                    const municipalityPD =
                        document.getElementById("municipality-pd");

                    const municipalityEL =
                        document.getElementById("municipality-el");

                    const municipalityApproval =
                        document.getElementById("municipality-approval");


                    layer.on({

                        mouseover: function(e) {

                            const municipio =
                                feature.properties.NOMGEO;

                            const municipalityData =
                                riskData[municipio];


                            municipalityName.textContent =
                                municipio;


                            if (municipalityData) {

                                municipalityPD.textContent =
                                    `${(municipalityData.predicted_pd * 100).toFixed(2)}%`;

                                municipalityEL.textContent =
                                    `$${municipalityData.expected_loss.toLocaleString()}`;

                                municipalityApproval.textContent =
                                    `${(municipalityData.approval_rate * 100).toFixed(0)}%`;

                            }

                            else {

                                municipalityPD.textContent =
                                    "NO DATA";

                                municipalityEL.textContent =
                                    "NO DATA";

                                municipalityApproval.textContent =
                                    "NO DATA";
                            }


                            municipalityCard.classList.add("active");


                            e.target.setStyle({

                                fillColor: "#155f97",

                                fillOpacity: 1
                            });

                        },


                        mouseout: function(e) {

                            municipalityCard.classList.remove("active");

                            const municipio =
                                feature.properties.NOMGEO;

                            const municipalityData =
                                riskData[municipio];

                            const risk =
                                municipalityData?.predicted_pd;


                            e.target.setStyle({

                                fillColor: getColor(risk),

                                fillOpacity: 0.8
                            });

                        }

                    });

                }

            }).addTo(map);


            map.fitBounds(
                geojsonLayer.getBounds()
            );

            map.zoomIn(0.3);

            map.panBy([0, 10]);

        });

}


/* =========================
   RISK COLORS
========================= */

function getColor(risk) {

    if (risk == null)
        return "#4f4f4f";

    if (risk >= 0.65) /* This is the threshold for very high risk */
        return "#4a0404";

    if (risk >= 0.50) /* This is the threshold for high risk */
        return "#8B0000";

    if (risk >= 0.35) /* This is the threshold for moderate risk */
        return "#ce6f02";

    if (risk >= 0.22) /* This is the threshold for low risk */
        return "#146a15";

    return "#3f634b";
}

/* =========================
   SCROLL ANIMATION
========================= */

const heroContent =
    document.querySelector(".hero-content");


window.addEventListener("scroll", () => {

    const scrollY = window.scrollY;

    const opacity =
        1 - scrollY / 50;

    heroContent.style.opacity =
        Math.max(opacity, 0);


    heroContent.style.transform =
        `translateY(calc(-50% - ${scrollY * 0.2}px))`;

});


fetch("dashboard_data.json")
    .then(response => response.json())
    .then(data => {

        dashboardData = data;

        loadDashboard();

    });

function loadDashboard() {

    const plotTheme = {
        paper_bgcolor: "rgba(239, 239, 239, 0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {
            color: "#ffffff",
            family: "Poppins"
        },
        xaxis: {
            gridcolor: "rgba(255,255,255,0.08)",
            zerolinecolor: "rgba(255,255,255,0.08)"
        },
        yaxis: {
            gridcolor: "rgba(255,255,255,0.08)",
            zerolinecolor: "rgba(255,255,255,0.08)"
        }
    };


    /* =========================
       ROC TRAIN
    ========================= */

    Plotly.newPlot("roc-train", [{
        x: dashboardData.roc_train.fpr,
        y: dashboardData.roc_train.tpr,
        mode: "lines",
        name: "Train ROC",
        line: {
            color: "#ffffff",
            width: 4
        }
    },
    {
        x: [0,1],
        y: [0,1],
        mode: "lines",
        name: "Baseline",
        line: {
            color: "#a40000",
            dash: "dash"
        }
    }], {
        ...plotTheme,
        title: "ROC Curve - Train",
        autosize: true,
        height: 520
    });


    /* =========================
       ROC TEST
    ========================= */

    Plotly.newPlot("roc-test", [{
        x: dashboardData.roc_test.fpr,
        y: dashboardData.roc_test.tpr,
        mode: "lines",
        name: "Test ROC",
        line: {
            color: "#fffefe",
            width: 4
        }
    },
    {
        x: [0,1],
        y: [0,1],
        mode: "lines",
        name: "Baseline",
        line: {
            color: "#b10000",
            dash: "dash"
        }
    }], {
        ...plotTheme,
        title: "ROC Curve - Test",
        autosize: true,
        height: 520
    });


    /* =========================
       CONFUSION MATRIX TRAIN
    ========================= */

    const cmTrain = dashboardData.cm_train;

    Plotly.newPlot("cm-train", [{
        z: cmTrain,
        type: "heatmap",
        colorscale: [[0, "#d6d6d6"],[1, "#101010"]],
        showscale: false,
        text: [
            [`TN<br>${cmTrain[0][0]}`, `FP<br>${cmTrain[0][1]}`],
            [`FN<br>${cmTrain[1][0]}`, `TP<br>${cmTrain[1][1]}`]
        ],
        texttemplate: "%{text}",
        textfont: {
            color: "#ffffff",
            size: 24,
            family: "Poppins"
        },
        hoverinfo: "skip"
    }], {
        ...plotTheme,
        title: "Confusion Matrix - Train",
        autosize: true,
        height: 520,
        xaxis: {
            title: "Predicted Label",
            tickvals: [0,1],
            ticktext: ["0","1"]
        },
        yaxis: {
            title: "True Label",
            tickvals: [0,1],
            ticktext: ["0","1"],
            autorange: "reversed"
        }
    });


    /* =========================
       CONFUSION MATRIX TEST
    ========================= */

    const cmTest = dashboardData.cm_test;

    Plotly.newPlot("cm-test", [{
        z: cmTest,
        type: "heatmap",
        colorscale: [[0, "#d6d6d6"],[1, "#101010"]],
        showscale: false,
        text: [
            [`TN<br>${cmTest[0][0]}`, `FP<br>${cmTest[0][1]}`],
            [`FN<br>${cmTest[1][0]}`, `TP<br>${cmTest[1][1]}`]
        ],
        texttemplate: "%{text}",
        textfont: {
            color: "#ffffff",
            size: 24,
            family: "Poppins"
        },
        hoverinfo: "skip"
    }], {
        ...plotTheme,
        title: "Confusion Matrix - Test",
        autosize: true,
        height: 520,
        xaxis: {
            title: "Predicted Label",
            tickvals: [0,1],
            ticktext: ["0","1"]
        },
        yaxis: {
            title: "True Label",
            tickvals: [0,1],
            ticktext: ["0","1"],
            autorange: "reversed"
        }
    });


    /* =========================
       DENSITY TRAIN
    ========================= */

    Plotly.newPlot("density-train", [

        {
            x: dashboardData.density_train.approved,
            type: "histogram",
            histnorm: "probability density",
            opacity: 0.55,
            name: "Approved",
            marker: {
                color: "#f3f0f0"
            }
        },

        {
            x: dashboardData.density_train.denied,
            type: "histogram",
            histnorm: "probability density",
            opacity: 0.55,
            name: "Denied",
            marker: {
                color: "#990000"
            }
        }

    ], {

        ...plotTheme,

        title: "Probability Density - Train",

        barmode: "overlay",

        autosize: true,

        height: 520,

        xaxis: {
            title: "Predicted Probability"
        },

        yaxis: {
            title: "Density"
        }

    });


    /* =========================
       DENSITY TEST
    ========================= */

    Plotly.newPlot("density-test", [

        {
            x: dashboardData.density_test.approved,
            type: "histogram",
            histnorm: "probability density",
            opacity: 0.55,
            name: "Approved",
            marker: {
                color: "#e2dcdc"
            }
        },

        {
            x: dashboardData.density_test.denied,
            type: "histogram",
            histnorm: "probability density",
            opacity: 0.55,
            name: "Denied",
            marker: {
                color: "#a40000"
            }
        }

    ], {

        ...plotTheme,

        title: "Probability Density - Test",

        barmode: "overlay",

        autosize: true,

        height: 520,

        xaxis: {
            title: "Predicted Probability"
        },

        yaxis: {
            title: "Density"
        }

    });



    /* =========================
    RISK BUCKET - TRAIN
    ========================= */

    Plotly.newPlot("risk-bucket-train", [{
        x: dashboardData.risk_bucket_train.labels,
        y: dashboardData.risk_bucket_train.values,
        type: "bar",
        marker: {
            color: [
                "#3f634b",
                "#d6a700",
                "#8B0000"
            ]
        }
    }], {
        ...plotTheme,
        title: "Risk Buckets - Train",
        autosize: true,
        height: 520,
        xaxis: {
            title: "Risk Bucket"
        },
        yaxis: {
            title: "Count"
        }
    });

    Plotly.newPlot("risk-bucket-test", [{
        x: dashboardData.risk_bucket_test.labels,
        y: dashboardData.risk_bucket_test.values,
        type: "bar",
        marker: {
            color: [
                "#3f634b",
                "#d6a700",
                "#8B0000"
            ]
        }
    }], {
        ...plotTheme,
        title: "Risk Buckets - Test",
        autosize: true,
        height: 520,
        xaxis: {
            title: "Risk Bucket"
        },
        yaxis: {
            title: "Count"
        }
    });

}