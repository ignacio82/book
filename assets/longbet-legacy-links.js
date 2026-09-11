// assets/longbet-legacy-links.js
// Client-side fragment router for sections and targets moved from longbet.html
(function() {
  var routes = {
    // Moved to longbet_decisions.html
    'sec-longbet-decision': 'longbet_decisions.html#sec-longbet-decision',
    'the-decision': 'longbet_decisions.html#sec-longbet-decision',
    'sec-longbet-multi': 'longbet_decisions.html#sec-longbet-multi',
    'when-the-decision-has-three-outcomes': 'longbet_decisions.html#sec-longbet-multi',
    'what-the-joint-model-shares-and-what-it-does-not': 'longbet_decisions.html#what-the-joint-model-shares-and-what-it-does-not',
    'the-supplemental-simulation': 'longbet_decisions.html#the-supplemental-simulation',
    'fitting-the-three-outcomes-together': 'longbet_decisions.html#fitting-the-three-outcomes-together',
    'the-three-condition-probability': 'longbet_decisions.html#the-three-condition-probability',
    'sampler-diagnostics-and-mixing': 'longbet_decisions.html#sampler-diagnostics-and-mixing',
    'what-the-joint-fit-bought-and-what-it-cost': 'longbet_decisions.html#what-the-joint-fit-bought-and-what-it-cost',
    'innovation-correlation': 'longbet_decisions.html#innovation-correlation',
    'fig-longbet-multi-heterogeneity': 'longbet_decisions.html#fig-longbet-multi-heterogeneity',
    'fig-longbet-multi-joint': 'longbet_decisions.html#fig-longbet-multi-joint',
    'tbl-longbet-multi': 'longbet_decisions.html#tbl-longbet-multi',

    // Moved to longbet_extensions.html
    'sec-longbet-forecast': 'longbet_extensions.html#sec-longbet-forecast',
    'projecting-past-the-end-of-the-study': 'longbet_extensions.html#sec-longbet-forecast',
    'sec-longbet-observational': 'longbet_extensions.html#sec-longbet-observational',
    'what-happens-without-the-randomization': 'longbet_extensions.html#sec-longbet-observational',
    'sec-longbet-limits': 'longbet_extensions.html#sec-longbet-limits',
    'practical-notes': 'longbet_extensions.html#sec-longbet-limits',
    'sec-longbet-baseline': 'longbet_extensions.html#sec-longbet-baseline',
    'where-a-units-own-level-comes-from': 'longbet_extensions.html#sec-longbet-baseline',
    'where-a-unit-s-own-level-comes-from': 'longbet_extensions.html#sec-longbet-baseline',
    'sec-longbet-serial': 'longbet_extensions.html#sec-longbet-serial',
    'the-error-term-is-independent-by-assumption': 'longbet_extensions.html#sec-longbet-serial',
    'parameter-guidance-and-design-choices': 'longbet_extensions.html#parameter-guidance-and-design-choices',
    'sec-longbet-timevarying': 'longbet_extensions.html#sec-longbet-timevarying',
    'covariates-that-change-over-the-panel': 'longbet_extensions.html#sec-longbet-timevarying',
    'sec-longbet-operating': 'longbet_extensions.html#sec-longbet-operating',
    'operating-characteristics': 'longbet_extensions.html#sec-longbet-operating',
    'sec-longbet-absorbing': 'longbet_extensions.html#sec-longbet-absorbing',
    'treatment-has-to-stay-on': 'longbet_extensions.html#sec-longbet-absorbing',
    'design-notes-for-the-experiment-itself': 'longbet_extensions.html#design-notes-for-the-experiment-itself',
    'fig-forecast': 'longbet_extensions.html#fig-forecast',
    'fig-promo': 'longbet_extensions.html#fig-promo'
  };

  function checkRoute() {
    try {
      var hash = window.location.hash;
      if (!hash || hash.length <= 1) return;
      var rawId = hash.substring(1);
      var targetId = decodeURIComponent(rawId);
      if (Object.prototype.hasOwnProperty.call(routes, targetId)) {
        var destination = routes[targetId];
        var parts = destination.split('#');
        var targetFile = parts[0];
        var targetAnchor = parts.length > 1 ? parts[1] : rawId;
        var search = window.location.search || '';
        var newUrl = targetFile + search + '#' + targetAnchor;
        window.location.replace(newUrl);
      }
    } catch (e) {
      // Malformed encoding or access error must not throw
    }
  }

  if (typeof window !== 'undefined') {
    checkRoute();
    window.addEventListener('hashchange', checkRoute);
  }
})();
