var pesosChart = new Chartist.Line(
    '.ct-chart',
    {
        labels: [0],

        series: [
            {
                name: 'Peso',
                data: [0]
            },
            {
                name: 'Linea del peso',
                data: [0]
            }
        ]
    },
    {
        series: {
            Peso: {
                lineSmooth: false
            },
            'Linea del peso': {
                lineSmooth: false
            }
        }
    }
);

var $chart = $('.ct-chart');

var $toolTip = $chart
    .append('<div class="tooltip"></div>')
    .find('.tooltip')
    .hide();

$chart.on('mouseenter', '.ct-point', function() {
    var $point = $(this),
        value = $point.attr('ct:value'),
        seriesName = $point.parent().attr('ct:series-name');

    $toolTip.html(seriesName + '<br>' + value).show();
});

$chart.on('mouseleave', '.ct-point', function() {
    $toolTip.hide();
});

$chart.on('mousemove', function(event) {
    $toolTip.css({
        left:
            (event.offsetX || event.originalEvent.layerX) -
            $toolTip.width() / 2 -
            5,
        top:
            (event.offsetY || event.originalEvent.layerY) -
            $toolTip.height() -
            20
    });
});

/**
 * PERCETRON
 */
var perceptron = new Perceptron();

perceptron
    .funcBack(funcBack)
    .addDatos([0, 0], 0)
    .addDatos([0, 1], 1)
    .addDatos([1, 0], 1)
    .addDatos([1, 1], 1)
    .aprender();

function funcBack(info) {
    var dataChart = pesosChart.data,
        labelInit = dataChart.labels[dataChart.labels.length - 1],
        pesos = info._pesos,
        lineaDePesos = [];

    for (var i = 1, len = pesos.length; i <= len; i++) {
        dataChart.labels.push(labelInit + i);
    }

    lineaDePesos.push(pesos[0], null, null);

    dataChart.series[0].data = dataChart.series[0].data.concat(pesos);
    dataChart.series[1].data = dataChart.series[1].data.concat(lineaDePesos);

    pesosChart.update({
        labels: dataChart.labels,
        series: dataChart.series
    });
}

console.log(perceptron, perceptron.procesar([1, 1, 1]));
