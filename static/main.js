const form = document.getElementById('carForm');
const resultEl = document.getElementById('result');

fetch('/metadata')
    .then(r => r.json())
    .then(meta => {
        Object.entries(meta).forEach(([col, val]) => {
            form.appendChild(buildField(col, val));
        });
    });

function buildField(col, val) {
    const wrap = document.createElement('div');
    wrap.className = 'flex flex-col gap-1';

    const label = document.createElement('span');
    label.textContent = col.replaceAll('_', ' ');
    label.className = 'text-sm text-slate-400';
    wrap.appendChild(label);

    // categoric -> dropdown/datalist
    if (Array.isArray(val)) {
        if (val.length > 300) {           // car_name có 4k value -> datalist
            const input = document.createElement('input');
            input.setAttribute('list', col + '_list');
            input.name = col;
            input.className = 'w-full px-3 py-2 bg-slate-700 rounded';
            wrap.appendChild(input);

            const dl = document.createElement('datalist');
            dl.id = col + '_list';
            val.forEach(opt => {
                const o = document.createElement('option');
                o.value = opt;
                dl.appendChild(o);
            });
            wrap.appendChild(dl);
        } else {
            const select = document.createElement('select');
            select.name = col;
            select.className = 'w-full px-3 py-2 bg-slate-700 rounded';
            select.innerHTML = '<option value="">-- chọn --</option>' +
                val.map(v => `<option value="${v}">${v}</option>`).join('');
            wrap.appendChild(select);
        }
    } else { // numeric
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.name = col;
        input.placeholder = val.toFixed(2);
        input.className = 'w-full px-3 py-2 bg-slate-700 rounded';
        wrap.appendChild(input);
    }
    return wrap;
}

document.getElementById('submitBtn').addEventListener('click', async e => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(form).entries());
    const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data),
    });
    const json = await res.json();
    resultEl.textContent = json.ok
        ? `Xe của bạn được định giá ≈ ${json.price.toLocaleString('vi-VN')} triệu ₫`
        : `❌ Lỗi: ${json.error}`;
});
