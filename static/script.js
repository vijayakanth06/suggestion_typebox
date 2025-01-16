$(document).ready(function () {
    
    $('#inputText').on('input', function () {
        let inputText = $(this).val().trim();

        if (inputText.length > 0) {
            $.ajax({
                type: 'POST',
                url: '/predict',
                contentType: 'application/json',
                data: JSON.stringify({ 'input_text': inputText, 'column_to_predict': 'name' }),
                success: function (response) {
                    let suggestions = response.suggestions;

                    $('#suggestions').html('');
                    if (suggestions.length > 0) {
                        suggestions.forEach(function (suggestion) {
                            let suggestionDiv = $('<div>').text(suggestion);
                            suggestionDiv.on('click', function () {
                                $('#inputText').val(suggestion.split(' (')[0]); 
                                $('#suggestions').html('');
                            });
                            $('#suggestions').append(suggestionDiv);
                        });
                    } else {
                        $('#suggestions').html('No similar words found.');
                    }
                },
                error: function () {
                    console.log('Error');
                }
            });
        } else {
            $('#suggestions').html('');
        }
    });
});
