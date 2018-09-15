// API for Google trend Data
const googleTrends = require('google-trends-api');
const fs = require('fs');
// First Argument is an Object with the Parameters to search for and second Argument is the callback

var startDate = new Date('2013-09-15');
var endDate = new Date('2018-09-14');

var toSearch = {
  keyword: "AMD",
  startTime: startDate, // a new Date object
  endTime: endDate, // a new Date object
};

googleTrends.interestOverTime(toSearch, (error, result) => {
    if(error) {
        console.log("Error found", error);
    }
    else {
        // getArray(result);
        // // console.log(result);
        let data = JSON.parse(result)

        let timeLineData = data.default.timelineData;

        let trendData = []
        timeLineData.forEach((element) => {
            for(i = 0; i < 7; i++){
                trendData.push(element.value[0])
            }
        })

        // console.log(trendData)
        // console.log(timeLineData)
        fs.writeFile('Trends.txt', trendData, (err) => {
            if(err) throw err;

            console.log('Write Successful');
        });
    }
});

// function getArray  (result) {
//     // var val = 0; var val = 0;
//     // var FinalValue = [];
//     // noinspection JSAnnotator
//     // result = result.default;
//
//     console.log(result;
//     // for (let value of result) {
//     //     console.log(value);
//     // }
// };
