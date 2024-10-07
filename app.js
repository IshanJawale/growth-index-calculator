fetch("https://24b9-35-203-149-139.ngrok-free.app/run-script", {
    
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      latitude: 19.95,
      longitude: 73.85,
      tree_name: "Mango",
      area_of_plantation: 10,
      manual_watering: 100,
      start_date: "2019-01-01",
      growth_index: -1
    }),
  })
  .then(async (res) => {
    const text_g = await res.text();
    console.log('Raw response:', text_g);
    try {
      const json = JSON.parse(text_g);
      console.log('Parsed JSON:', json);
    } catch (error) {
      console.error('Error parsing JSON:', error);
    }
  })
  .catch((err) => {
    console.error('Error:', err);
  });
  