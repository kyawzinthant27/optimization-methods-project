// Injected before page scripts. Same instrumentation in every measured condition.
(() => {
  performance.setResourceTimingBufferSize(1000);
  const images = [];
  const errors = [];
  const lcpEntries = [];
  const visibilityChanges = [];
  const pathOf = (url) => new URL(url, location.href).pathname;
  const observeLcp = (entries) => {
    for (const entry of entries) {
      const element = entry.element;
      lcpEntries.push({
        start_time_ms: entry.startTime,
        render_time_ms: entry.renderTime,
        load_time_ms: entry.loadTime,
        size: entry.size,
        url: entry.url || null,
        tag: element?.tagName ?? null,
        id: element?.id ?? null,
        class_name: typeof element?.className === 'string' ? element.className : null,
        text: element?.textContent?.trim().slice(0, 200) ?? null,
        image_id: element?.closest('figure')?.dataset.imageId ?? null,
      });
    }
  };
  let lcpObserver;

  try {
    lcpObserver = new PerformanceObserver((list) => observeLcp(list.getEntries()));
    lcpObserver.observe({
      type: 'largest-contentful-paint',
      buffered: true,
    });
  } catch (error) {
    errors.push(`LCP observer: ${error.message}`);
  }

  addEventListener('visibilitychange', () =>
    visibilityChanges.push({
      state: document.visibilityState,
      at_ms: performance.now(),
    }),
  );
  document.addEventListener(
    'error',
    (event) => {
      const image = event.target;

      if (image instanceof HTMLImageElement)
        errors.push(`Image failed: ${pathOf(image.src)}`);
    },
    true,
  );
  document.addEventListener(
    'load',
    (event) => {
      const image = event.target;

      if (!(image instanceof HTMLImageElement) || !pathOf(image.src).startsWith('/images/'))
        return;

      const row = {
        path: pathOf(image.src),
        load_event_ms: performance.now(),
        decode_started_ms: null,
        decode_ready_ms: null,
        decode_error: null,
      };
      images.push(row);
      // Calling decode after load does not alter fetch scheduling. It forces readiness
      // for offscreen images too. Elapsed time is not pure CPU decoding time.
      row.decode_started_ms = performance.now();
      image
        .decode()
        .then(() => {
          row.decode_ready_ms = performance.now();
        })
        .catch((error) => {
          row.decode_error = error.message;
          errors.push(`Decode failed: ${row.path}`);
        });
    },
    true,
  );
  window.__imageBenchmark = {
    images,
    errors,
    done: () =>
      (images.length === 14 &&
        images.every((i) => i.decode_ready_ms !== null || i.decode_error)) ||
      errors.length > 0,
    snapshot: () => {
      if (lcpObserver) observeLcp(lcpObserver.takeRecords());

      return {
        time_origin_ms: performance.timeOrigin,
        observation_end_ms: performance.now(),
        visibility: document.visibilityState,
        visibility_changes: visibilityChanges,
        images,
        errors,
        lcp_entries: lcpEntries,
        fcp_ms:
          performance.getEntriesByName('first-contentful-paint')[0]?.startTime ?? null,
        resources: performance.getEntriesByType('resource').map((e) => ({
          path: pathOf(e.name),
          initiator_type: e.initiatorType,
          start_ms: e.startTime,
          request_start_ms: e.requestStart,
          response_start_ms: e.responseStart,
          response_end_ms: e.responseEnd,
          encoded_body_bytes: e.encodedBodySize,
          decoded_body_bytes: e.decodedBodySize,
          transfer_bytes: e.transferSize,
          protocol: e.nextHopProtocol,
          delivery_type: e.deliveryType ?? null,
        })),
        navigation: performance.getEntriesByType('navigation')[0]?.toJSON() ?? null,
        gallery: [...document.querySelectorAll('#gallery img')].map((i) => ({
          path: pathOf(i.currentSrc),
          complete: i.complete,
          natural_width: i.naturalWidth,
          natural_height: i.naturalHeight,
          width_attribute: Number(i.getAttribute('width')),
          height_attribute: Number(i.getAttribute('height')),
          loading: i.loading,
          decoding: i.decoding,
          fetch_priority: i.fetchPriority,
          srcset: i.srcset,
        })),
        frames: [...document.querySelectorAll('.image-frame')].map((e) => {
          const r = e.getBoundingClientRect();

          return {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
          };
        }),
        page_ready: document.documentElement.dataset.ready,
        feasible: document.body.dataset.feasible,
        batch_status: document.querySelector('#batch-status')?.textContent,
        scroll_x: scrollX,
        scroll_y: scrollY,
      };
    },
  };
})();
