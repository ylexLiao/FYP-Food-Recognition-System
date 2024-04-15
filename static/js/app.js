const { createApp, ref, onMounted } = Vue;

createApp({
    setup() {
        // ref 用于响应性地存储图片文件和识别结果
        const imageFile = ref(null);
        const result = ref('');
        const description = ref('');
        const uploadedImg = ref(null);


        onMounted(() => {
            const fileUpload = document.getElementById('file-upload');
            fileUpload.addEventListener('change', onFileChange);
        });

        const onFileChange = e => {
            const files = e.target.files;
            if (files.length > 0) {
                // 存储文件而不是文件的数据URL
                imageFile.value = files[0];

                // 创建一个 URL 并将其赋值给 img 标签的 src 属性以预览图片
                const reader = new FileReader();
                reader.onload = (e) => {
                    // 这里的 image 是用来预览图片的
                    document.getElementById('uploaded-img').src = e.target.result;
                    document.getElementById('uploaded-img').style.display = 'block';
                };
                reader.readAsDataURL(files[0]);
            }
        };

        //  onMounted(() => {
        //     const fileUpload = document.getElementById('file-upload');
        //     fileUpload.addEventListener('change', onFileChange);
        // });

        // result.value = "Predicting...";

        const recognizeFood = async () => {
            // Initialize a new FormData object
            const formData = new FormData();
        
            // Append the file under 'image' key as a Blob
            formData.append('image', imageFile.value);
        
            // Setting the 'Content-Type' to 'multipart/form-data' is not required, as the browser will set it automatically
            const requestOptions = {
                method: 'POST',
                body: formData,
            };
        
            try {
                const response = await fetch('/recognize', requestOptions);
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const data = await response.json();
                // Log the response data to see what is being received
                console.log(data);
                
                result.value = data.result;
                description.value = data.description; 

                console.log(result.value);

            } catch (error) {
                console.error('Error:', error);
            }
        };

        // 保持返回的属性和方法
        return { imageFile, result, description, onFileChange, recognizeFood };
    },

    data() {
        return {
          message: 'Hello Vue!'
        };
       
      },
      compilerOptions:{
        delimiters: ['[[', ']]'] // 自定义界定符
    }
    
}
).mount('#app');
app.config.compilerOptions.delimiters = ['[[', ']]'];
