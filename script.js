document.getElementById("switch-en").addEventListener("click", function () {
    document.querySelector(".en").style.display = "block";
    document.querySelector(".cn").style.display = "none";
  });
  
  document.getElementById("switch-cn").addEventListener("click", function () {
    document.querySelector(".en").style.display = "none";
    document.querySelector(".cn").style.display = "block";
  });
  