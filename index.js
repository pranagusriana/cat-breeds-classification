const MODEL_PATH = 'cat_breeds_model/model.json';
const IMAGE_SIZE = 299;

const CLASS_LABELS = {0: 'Abyssinian',
 1: 'American Bobtail',
 2: 'American Curl',
 3: 'American Shorthair',
 4: 'American Wirehair',
 5: 'Balinese',
 6: 'Bengal',
 7: 'Birman',
 8: 'Bombay',
 9: 'British Shorthair',
 10: 'Burmese',
 11: 'Burmilla',
 12: 'Chartreux',
 13: 'Cornish Rex',
 14: 'Devon Rex',
 15: 'Domestic',
 16: 'Egyptian Mau',
 17: 'Exotic',
 18: 'Havana Brown',
 19: 'Himalaya',
 20: 'Japanese Bobtail',
 21: 'Khao Manee',
 22: 'Korat',
 23: 'LaPerm',
 24: 'Lykoi',
 25: 'Maine Coon',
 26: 'Manx',
 27: 'Munchkin',
 28: 'Norwegian Forest',
 29: 'Ocicat',
 30: 'Oriental',
 31: 'Persian',
 32: 'RagaMuffin',
 33: 'Ragdoll',
 34: 'Russian Blue',
 35: 'Scottish Fold',
 36: 'Selkirk Rex',
 37: 'Siamese',
 38: 'Siberian',
 39: 'Singapura',
 40: 'Snowshoe',
 41: 'Somali',
 42: 'Sphynx',
 43: 'Tonkinese',
 44: 'Toybob',
 45: 'Turkish Angora',
 46: 'Turkish Van'}

var model;

const selected_image = document.getElementById('selected-image');
const image_selector = document.getElementById('image-selector');
const predict_result = document.getElementById('prediction-result');

async function initialize() {
    model = await tf.loadLayersModel(MODEL_PATH);
}

initialize();

async function predict(image) {
    const startTime = performance.now();
    const res = tf.tidy(() => {
        var img = tf.browser.fromPixels(image).toFloat();
        // Resize image
        img = img.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        // Normalize image
        const offset = tf.scalar(255);
        img = img.div(offset);
        return model.predict(img);
    });

    var prediction =  Array.from(res.dataSync());
    const totalTime = performance.now() - startTime;

    // Process to get n result
    var class_idx = [];
    var prediction_temp = Array(...prediction);
    var n_res = 3;
    for (var i = 0; i < n_res; i++){
        var max_value = Math.max(...prediction_temp);
        var idx_max = prediction.indexOf(max_value);
        class_idx.push(idx_max);
        prediction_temp.splice(prediction_temp.indexOf(max_value), 1);
    }

    // Process for return
    objReturn = [];

    var strRet = "" + n_res + " Possible Cat Breeds:\n"
    for(var n = 0; n < n_res; n++){
        var obj = {
            class: CLASS_LABELS[class_idx[n]],
            confident: prediction[class_idx[n]],
        };
        objReturn.push(obj);
        strRet = strRet + '\xa0'.repeat(4) + (n+1) + ". " + obj.class + " (" + obj.confident * 100 + "%)\n"
    }

    // Show result to HTML
    predict_result.innerText = strRet + "Time: " + totalTime + " ms.\n"
}

image_selector.addEventListener('change', e => {
    let files = e.target.files;

    let reader = new FileReader();
    reader.onload = er => {
        // Load image
        let img = new Image();
        img.src = er.target.result;
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
        
        // Tampilin image di html
        selected_image.src = er.target.result;
        selected_image.width = IMAGE_SIZE;
        selected_image.height = IMAGE_SIZE;

        // Predict
        predict_result.innerText = 'Predicting ...';
        img.onload = () => predict(img);
    };

    reader.readAsDataURL(files[0]);
});
