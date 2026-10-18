/**
 * Custom JS for Abilasch Vijikaran's portfolio
 * 1. Makes every external link (and the CV download) open in a new tab.
 * 2. Makes the contact form work using Web3Forms (https://web3forms.com).
 *    Uses the template's existing .loading / .error-message / .sent-message elements.
 */
(function () {
  "use strict";

  /* ---------- 1. All links open in new tabs ---------- */
  document.querySelectorAll('a[href^="http"], a[href$=".pdf"]').forEach(function (link) {
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
  });

  /* ---------- 2. Working contact form ---------- */
  const form = document.getElementById("contact-form");
  if (!form) return;

  const loading = form.querySelector(".loading");
  const errorMsg = form.querySelector(".error-message");
  const sentMsg = form.querySelector(".sent-message");

  function showError(text) {
    loading.classList.remove("d-block");
    errorMsg.textContent = text;
    errorMsg.classList.add("d-block");
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();

    loading.classList.add("d-block");
    errorMsg.classList.remove("d-block");
    sentMsg.classList.remove("d-block");

    try {
      const response = await fetch(form.action, {
        method: "POST",
        headers: { Accept: "application/json" },
        body: new FormData(form)
      });
      const result = await response.json();

      if (response.ok && result.success) {
        loading.classList.remove("d-block");
        sentMsg.classList.add("d-block");
        form.reset();
      } else {
        showError(result.message || "Sorry, your message could not be sent. Please try again.");
      }
    } catch (err) {
      showError("Sorry, your message could not be sent. Please check your connection and try again.");
    }
  });
})();
