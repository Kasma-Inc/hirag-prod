from sqlalchemy import TextClause, text

search_by_search_list: TextClause = text(
    """CREATE OR REPLACE FUNCTION search_by_search_list (original_text_normalized TEXT, translation_normalized TEXT, original_token_list CHARACTER VARYING[], original_token_start_index_list INTEGER[], original_token_end_index_list INTEGER[], translation_token_list CHARACTER VARYING[], translation_token_start_index_list INTEGER[], translation_token_end_index_list INTEGER[], search_item_list_original CHARACTER VARYING[], search_item_list CHARACTER VARYING[])
RETURNS TABLE(
    original_text_search_result TEXT,
    translation_search_result TEXT
) AS $$
    from typing import List, Dict, Tuple, Optional, Literal
    from rapidfuzz import fuzz
    from rapidfuzz.distance import ScoreAlignment

    fuzzy_match_start_index_list_original: Optional[List[int]] = None
    fuzzy_match_end_index_list_original: Optional[List[int]] = None
    fuzzy_match_start_index_list_translation: Optional[List[int]] = None
    fuzzy_match_end_index_list_translation: Optional[List[int]] = None

    original_text_search_result: Optional[str] = None
    translation_search_result: Optional[str] = None

    def fuzzy_match(
        text_normalized: str, search_list: List[str]
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        fuzzy_match_index_start_list: Optional[List[int]] = []
        fuzzy_match_index_end_list: Optional[List[int]] = []
        queue: List[Tuple[str, int]] = [(text_normalized, 0)]
        while len(queue) > 0:
            text, start_index = queue.pop(0)
            for search in search_list:
                if fuzz.ratio(text, search) > 95:
                    fuzzy_match_index_start_list.append(start_index)
                    fuzzy_match_index_end_list.append(start_index + len(text))
                    break
                elif len(text) >= len(search):
                    match_result: Optional[ScoreAlignment] = fuzz.partial_ratio_alignment(
                        text, search, score_cutoff=95
                    )
                    if match_result is not None:
                        fuzzy_match_index_start_list.append(start_index + match_result.src_start)
                        fuzzy_match_index_end_list.append(start_index + match_result.src_end)
                        if match_result.src_start > 0:
                            queue.append(
                                (
                                    text[: match_result.src_start],
                                    start_index,
                                )
                            )
                        if match_result.src_end < len(text):
                            queue.append(
                                (
                                    text[match_result.src_end :],
                                    start_index + match_result.src_end,
                                )
                            )
        if len(fuzzy_match_index_start_list) > 0:
            return fuzzy_match_index_start_list, fuzzy_match_index_end_list
        else:
            return None, None

    def get_token_index(
        char_index: int,
        char_type: Literal["original", "translation"]
    ) -> Tuple[int, bool]:
        token_start_index_list: List[int] = original_token_start_index_list if char_type == "original" else translation_token_start_index_list
        token_end_index_list: List[int] = original_token_end_index_list if char_type == "original" else translation_token_end_index_list
        left_index: int = 0
        right_index: int = len(token_start_index_list)
        while left_index < right_index:
            mid_index: int = (left_index + right_index) // 2
            if (char_index >= token_start_index_list[mid_index]) and (
                char_index < token_end_index_list[mid_index]
            ):
                return mid_index, True
            elif token_start_index_list[mid_index] > char_index:
                right_index = mid_index
            else:
                left_index = mid_index + 1
        return left_index, False


    def bold_matched_text(
        text_type: Literal["original", "translation"],
    ) -> None:
        fuzzy_match_start_index_list: Optional[List[int]] = fuzzy_match_start_index_list_original if text_type == "original" else fuzzy_match_start_index_list_translation
        fuzzy_match_end_index_list: Optional[List[int]] = fuzzy_match_end_index_list_original if text_type == "original" else fuzzy_match_end_index_list_translation
        token_list: List[str] = original_token_list if text_type == "original" else translation_token_list
        for i, fuzzy_match_start_index in enumerate(
            fuzzy_match_start_index_list
        ):
            start, _ = get_token_index(
                fuzzy_match_start_index,
                text_type
            )
            end, in_token = get_token_index(
                fuzzy_match_end_index_list[i] - 1,
                text_type
            )
            if in_token:
                end += 1
            for j in range(start, end):
                if "<mark>" not in token_list[j]:
                    token_list[j] = f"<mark>{token_list[j]}</mark>"

        for j in range(len(token_list) - 1):
            if token_list[j].endswith(
                "</mark>"
            ) and token_list[j + 1].startswith("<mark>"):
                token_list[j] = token_list[j][:-7]
                token_list[j + 1] = token_list[j + 1][6:]

    def simplify_search_result(
        text_type: Literal["original", "translation"],
    ) -> Optional[str]:
        text_normalized: str = original_text_normalized if text_type == "original" else translation_normalized
        token_list: List[str] = original_token_list if text_type == "original" else translation_token_list
        token_start_index_list: List[int] = original_token_start_index_list if text_type == "original" else translation_token_start_index_list
        token_end_index_list: List[int] = original_token_end_index_list if text_type == "original" else translation_token_end_index_list
        matched_index_tuple_list: List[Tuple[int, int]] = []
        for j, word in enumerate(token_list):
            if word.startswith("<mark>") and word.endswith("</mark>"):
                matched_index_tuple_list.append((j, j + 1))
            elif word.startswith("<mark>"):
                matched_index_tuple_list.append(
                    (j, len(token_list))
                )
            elif word.endswith("</mark>"):
                matched_index_tuple_list[-1] = (matched_index_tuple_list[-1][0], j + 1)

        output: str = ""
        last_match_end: int = -1
        last_end: int = -1
        for matched_index_tuple in matched_index_tuple_list:
            match_start: int = token_start_index_list[
                matched_index_tuple[0]
            ]
            match_end: int = token_end_index_list[
                matched_index_tuple[1] - 1
            ]
            start: int = token_start_index_list[
                max(0, matched_index_tuple[0] - 3)
            ]
            end: int = token_end_index_list[
                min(
                    len(token_list),
                    matched_index_tuple[1] + 3,
                )
                - 1
            ]
            if (start != 0) and (start > last_end):
                output += "..."
            elif start < last_match_end:
                output = output[: last_match_end - last_end]
                start = last_match_end
            elif start < last_end:
                output = output[: start - last_end]
            output += (
                text_normalized[start:match_start]
                + f"<mark>{text_normalized[match_start:match_end]}</mark>"
                + text_normalized[match_end:end]
            )
            last_match_end = match_end
            last_end = end
        if len(output) > 0:
            if last_end < len(text_normalized):
                output += "..."
            return output
        else:
            return None


    fuzzy_match_start_index_list_original, fuzzy_match_end_index_list_original = fuzzy_match(
        original_text_normalized, search_item_list_original
    )
    if len(search_item_list) > 0:
        if fuzzy_match_start_index_list_original is not None:
            search_result_start_index, search_result_end_index = fuzzy_match(
                original_text_normalized, search_item_list
            )
            if search_result_start_index is not None:
                fuzzy_match_start_index_list_original += search_result_start_index
                fuzzy_match_end_index_list_original += search_result_end_index
        else:
            fuzzy_match_start_index_list_translation, fuzzy_match_end_index_list_translation = fuzzy_match(
                translation_normalized, search_item_list
            )

    if fuzzy_match_start_index_list_original is not None:
        bold_matched_text(
            "original",
        )
        original_text_search_result = simplify_search_result("original")

    if fuzzy_match_start_index_list_translation is not None:
        bold_matched_text(
            "translation",
        )
        translation_search_result = simplify_search_result("translation")


    yield original_text_search_result, translation_search_result
$$ LANGUAGE plpython3u;
"""
)
