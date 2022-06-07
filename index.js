const MODEL_PATH = 'cat_breeds_model/model.json';
const IMAGE_SIZE = 150;

const CLASS_LABELS = {
 0: 'Abyssinian',
 1: 'American Bobtail',
 2: 'American Curl',
 3: 'American Shorthair',
 4: 'American Wirehair',
 5: 'Applehead Siamese',
 6: 'Balinese',
 7: 'Bengal',
 8: 'Birman',
 9: 'Bombay',
 10: 'British Shorthair',
 11: 'Burmese',
 12: 'Burmilla',
 13: 'Calico',
 14: 'Canadian Hairless',
 15: 'Chartreux',
 16: 'Chausie',
 17: 'Chinchilla',
 18: 'Cornish Rex',
 19: 'Cymric',
 20: 'Devon Rex',
 21: 'Dilute Calico',
 22: 'Dilute Tortoiseshell',
 23: 'Domestic Long Hair',
 24: 'Domestic Medium Hair',
 25: 'Domestic Short Hair',
 26: 'Egyptian Mau',
 27: 'Exotic Shorthair',
 28: 'Extra-Toes Cat - Hemingway Polydactyl',
 29: 'Havana',
 30: 'Himalayan',
 31: 'Japanese Bobtail',
 32: 'Javanese',
 33: 'Korat',
 34: 'LaPerm',
 35: 'Maine Coon',
 36: 'Manx',
 37: 'Munchkin',
 38: 'Nebelung',
 39: 'Norwegian Forest Cat',
 40: 'Ocicat',
 41: 'Oriental Long Hair',
 42: 'Oriental Short Hair',
 43: 'Oriental Tabby',
 44: 'Persian',
 45: 'Pixiebob',
 46: 'Ragamuffin',
 47: 'Ragdoll',
 48: 'Russian Blue',
 49: 'Scottish Fold',
 50: 'Selkirk Rex',
 51: 'Siamese',
 52: 'Siberian',
 53: 'Silver',
 54: 'Singapura',
 55: 'Snowshoe',
 56: 'Somali',
 57: 'Sphynx - Hairless Cat',
 58: 'Tabby',
 59: 'Tiger',
 60: 'Tonkinese',
 61: 'Torbie',
 62: 'Tortoiseshell',
 63: 'Turkish Angora',
 64: 'Turkish Van',
 65: 'Tuxedo',
 66: 'York Chocolate'
}

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
