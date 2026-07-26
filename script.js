const shopperTable = document.getElementById("shopperTable");

const shopperCount = document.getElementById("totalShoppers");
const attentionTime = document.getElementById("attentionTime");
const shopperSegment = document.getElementById("shopperSegment");

async function loadShoppers() {

    const response = await fetch("http://127.0.0.1:8000/analytics/shoppers?cache="+Date.now());
    const data = await response.json();

    if (!data || Object.keys(data).length === 0) {
    return;
}

    console.log(data);

    shopperTable.innerHTML = "";

    let total = 0;
    let latestAttention = 0;
    let latestSegment = "-";

    for (const id in data) {

        const shopper = data[id];

        total++;

        latestAttention = shopper.attention_time;
        latestSegment = shopper.segment;

        shopperTable.innerHTML += `
        <tr>
            <td>${shopper.shopper_id}</td>
            <td>${shopper.attention_time.toFixed(2)} sec</td>
            <td>${shopper.segment}</td>
            <td>${shopper.updated_at}</td>
        </tr>
        `;
    }

    shopperCount.innerText = total;
    attentionTime.innerText = latestAttention.toFixed(2) + " sec";
    shopperSegment.innerText = latestSegment;
}

loadShoppers();

setInterval(() => {
    loadShoppers();
}, 5000);
