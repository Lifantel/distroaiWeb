(function () {
  "use strict";

  var DISTRO_NAMES = {
    0: "Linux Mint",
    1: "Ubuntu",
    2: "Fedora",
    3: "CachyOS",
    4: "Arch Linux",
    5: "openSUSE",
    6: "Pop!_OS",
    7: "Kali Linux",
    8: "Pardus",
    9: "Tails"
  };

  var QUESTIONS = [
    "Daha önce Linux veya terminal kullandın mı?",
    "Bilgisayarın güncel ve donanımı güçlü mü?",
    "Kararlılıktan çok en yeni güncellemeleri almak senin için daha mı önemli?",
    "Bu sistemi yoğun şekilde oyun veya performans odaklı işler için mi kuracaksın?",
    "Bilgisayarında NVIDIA ekran kartı var mı?",
    "Sistemin çok az RAM ve CPU tüketmesi senin için kritik mi?",
    "Macera ve özelleştirme istiyor musun?",
    "Büyük bir şirket desteği olmasını ister misin?",
    "Siber güvenlik veya sızma testi yapmak istiyor musun?",
    "Windows'a benzer bir deneyim ister misin?",
    "Sistemin USB üzerinden çalışabilmesini ister misin?",
    "Sağlıklı bir yaşam istiyor musun?",
    "Daha mobil ve laptop uyumlu bir arayüz ister misin?",
    "Tamamen anonimlik ve iz bırakmamayı mı amaçlıyorsun? (USB üzerinden çalışan Tails'e yönlendirir)"
  ];

  var SCALE_MAP = { 1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0 };

  var BOOT_LINES = [
    { text: "$ ./distroai --init", cls: "" },
    { text: "torch ağırlıkları belleğe yükleniyor ...", cls: "dim" },
    { text: "katmanlar: 14 → 32 → 16 → 10", cls: "dim" },
    { text: "doğrulama başarımı: %94.8", cls: "ok" },
    { text: "sunucu bağlantısı yok, çıkarım cihazda çalışacak", cls: "dim" },
    { text: "hazır.", cls: "ok" }
  ];

  // ---------- Forward pass (matches DistroAI: Linear-ReLU-Linear-ReLU-Linear) ----------
  function linear(x, layer) {
    var w = layer.w, b = layer.b;
    var out = new Array(w.length);
    for (var i = 0; i < w.length; i++) {
      var sum = b[i];
      var row = w[i];
      for (var j = 0; j < row.length; j++) sum += row[j] * x[j];
      out[i] = sum;
    }
    return out;
  }

  function relu(x) {
    return x.map(function (v) { return v > 0 ? v : 0; });
  }

  function softmax(x) {
    var max = Math.max.apply(null, x);
    var exps = x.map(function (v) { return Math.exp(v - max); });
    var sum = exps.reduce(function (a, b) { return a + b; }, 0);
    return exps.map(function (v) { return v / sum; });
  }

  function predict(inputVector) {
    var h1 = relu(linear(inputVector, MODEL_WEIGHTS[0]));
    var h2 = relu(linear(h1, MODEL_WEIGHTS[1]));
    var logits = linear(h2, MODEL_WEIGHTS[2]);
    return softmax(logits);
  }

  // ---------- DOM ----------
  var bootScreen = document.getElementById("boot");
  var bootLog = document.getElementById("boot-log");
  var bootSkip = document.getElementById("boot-skip");
  var introScreen = document.getElementById("intro");
  var startBtn = document.getElementById("start-btn");
  var quizScreen = document.getElementById("quiz");
  var resultScreen = document.getElementById("result");

  var qCurrentEl = document.getElementById("q-current");
  var qTotalEl = document.getElementById("q-total");
  var progressFill = document.getElementById("progress-fill");
  var qTextEl = document.getElementById("q-text");
  var scaleRow = document.getElementById("scale-row");
  var backBtn = document.getElementById("back-btn");

  var resultName = document.getElementById("result-name");
  var resultPct = document.getElementById("result-pct");
  var resultBars = document.getElementById("result-bars");
  var restartBtn = document.getElementById("restart-btn");

  qTotalEl.textContent = QUESTIONS.length;

  var answers = new Array(QUESTIONS.length).fill(null);
  var currentQ = 0;
  var bootTimer = null;

  function showScreen(el) {
    [bootScreen, introScreen, quizScreen, resultScreen].forEach(function (s) {
      s.classList.add("hidden");
    });
    el.classList.remove("hidden");
  }

  // ---------- Boot sequence ----------
  function runBoot() {
    var i = 0;
    bootLog.textContent = "";
    function step() {
      if (i >= BOOT_LINES.length) {
        bootTimer = setTimeout(finishBoot, 500);
        return;
      }
      var line = BOOT_LINES[i];
      var span = document.createElement("div");
      if (line.cls) span.className = line.cls;
      span.textContent = line.text;
      bootLog.appendChild(span);
      i++;
      bootTimer = setTimeout(step, 260);
    }
    step();
  }

  function finishBoot() {
    clearTimeout(bootTimer);
    showScreen(introScreen);
  }

  bootSkip.addEventListener("click", finishBoot);

  // ---------- Intro ----------
  startBtn.addEventListener("click", function () {
    currentQ = 0;
    answers.fill(null);
    showScreen(quizScreen);
    renderQuestion();
  });

  // ---------- Quiz ----------
  function renderQuestion() {
    qCurrentEl.textContent = currentQ + 1;
    progressFill.style.width = ((currentQ) / QUESTIONS.length * 100) + "%";
    progressFill.parentElement.setAttribute("aria-valuenow", currentQ);
    qTextEl.textContent = QUESTIONS[currentQ];
    backBtn.disabled = currentQ === 0;

    var buttons = scaleRow.querySelectorAll(".scale-btn");
    buttons.forEach(function (btn) {
      var val = parseInt(btn.dataset.val, 10);
      btn.classList.toggle("selected", answers[currentQ] === val);
    });
  }

  scaleRow.addEventListener("click", function (e) {
    var btn = e.target.closest(".scale-btn");
    if (!btn) return;
    var val = parseInt(btn.dataset.val, 10);
    answers[currentQ] = val;
    progressFill.style.width = ((currentQ + 1) / QUESTIONS.length * 100) + "%";

    setTimeout(function () {
      if (currentQ < QUESTIONS.length - 1) {
        currentQ++;
        renderQuestion();
      } else {
        computeResult();
      }
    }, 160);
  });

  backBtn.addEventListener("click", function () {
    if (currentQ > 0) {
      currentQ--;
      renderQuestion();
    }
  });

  // ---------- Result ----------
  function computeResult() {
    var vector = answers.map(function (v) { return SCALE_MAP[v]; });
    var probs = predict(vector);

    var ranked = Object.keys(DISTRO_NAMES).map(function (idx) {
      return { idx: parseInt(idx, 10), name: DISTRO_NAMES[idx], p: probs[idx] };
    }).sort(function (a, b) { return b.p - a.p; });

    resultName.textContent = ranked[0].name;
    resultPct.textContent = "%" + (ranked[0].p * 100).toFixed(1);

    resultBars.innerHTML = "";
    ranked.forEach(function (item, i) {
      var row = document.createElement("div");
      row.className = "result-row" + (i === 0 ? " top" : "");

      var name = document.createElement("span");
      name.className = "result-row-name";
      name.textContent = item.name;

      var track = document.createElement("span");
      track.className = "result-row-track";
      var fill = document.createElement("span");
      fill.className = "result-row-fill";
      track.appendChild(fill);

      var pct = document.createElement("span");
      pct.className = "result-row-pct";
      pct.textContent = "%" + (item.p * 100).toFixed(1);

      row.appendChild(name);
      row.appendChild(track);
      row.appendChild(pct);
      resultBars.appendChild(row);

      setTimeout(function () {
        fill.style.width = (item.p * 100) + "%";
      }, 30 + i * 40);
    });

    showScreen(resultScreen);
  }

  restartBtn.addEventListener("click", function () {
    currentQ = 0;
    answers.fill(null);
    showScreen(introScreen);
  });

  // ---------- Init ----------
  runBoot();
})();
