document.addEventListener("DOMContentLoaded", function () {

    const imageInput = document.getElementById("imageInput");
    const previewImg = document.getElementById("previewImg");
    const statusText = document.getElementById("uploadStatus");
    const form = document.querySelector("form");

    imageInput.addEventListener("change", function () {
        const file = this.files[0];

        if (file) {
            // Preview image
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImg.src = e.target.result;
                previewImg.style.display = "block";
            };
            reader.readAsDataURL(file);

            statusText.style.color = "#9adfff";
            statusText.innerText =
                "✅ Image successfully uploaded: " + file.name;
        }
    });

    // 🔥 Predict button validation
    window.validateAndSubmit = function () {
        if (!imageInput.files || imageInput.files.length === 0) {
            statusText.style.color = "#ff6b6b";
            statusText.innerText = "⚠️ Please upload an image before prediction";
            return;
        }

        // If image exists → submit form
        form.submit();
    };

});
function toggleMenu() {
    const menu = document.getElementById("dropdownMenu");

    if (menu.style.display === "flex") {
        menu.style.display = "none";
    } else {
        menu.style.display = "flex";
    }
}

function openMenu() {
    document.getElementById("sideMenu").classList.add("active");
    document.getElementById("overlay").classList.add("active");
}

function closeMenu() {
    document.getElementById("sideMenu").classList.remove("active");
    document.getElementById("overlay").classList.remove("active");
}
