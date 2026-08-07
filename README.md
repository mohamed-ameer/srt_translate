
1. install py packages
```bash
pip install google-generativeai openai
```
2. get your apikey => [Google Gemini API Key](https://aistudio.google.com/api-keys)
3. download the arabic subtitle in srt and past it in the `input_ar.srt` file => [DownSub Site](https://downsub.com/)
4. run the following:
```bash
export GEMINI_API_KEY=your_key_here
python srt_translate.py --list-models
python srt_translate.py input_ar.srt -o output_id.srt --lang Indonesian --provider gemini --model gemini-3.5-flash-lite --batch-size 1000
```