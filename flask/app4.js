fetch("https://6620-35-245-131-176.ngrok-free.app/run-script", {
    
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      marks_obtained: 20,
      total_marks: 150,
      percentage: -1
    }),
  })
  .then(async (res) => {
    const text = await res.text();
    console.log('Raw response:', text);
    try {
      const json = JSON.parse(text);
      console.log('Parsed JSON:', json);
    } catch (error) {
      console.error('Error parsing JSON:', error);
    }
  })
  .catch((err) => {
    console.error('Error:', err);
  });
  