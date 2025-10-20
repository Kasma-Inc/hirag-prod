from sqlalchemy import TextClause, text

search_by_search_list: TextClause = text(
    """CREATE OR REPLACE FUNCTION search_by_search_list (search_item_list_original CHARACTER VARYING[], original_text_normalized TEXT, original_token_list CHARACTER VARYING[], original_token_start_index_list INTEGER[], original_token_end_index_list INTEGER[], search_item_list CHARACTER VARYING[], translation_normalized TEXT, translation_token_list CHARACTER VARYING[], translation_token_start_index_list INTEGER[], translation_token_end_index_list INTEGER[])
RETURNS TEXT AS $$
    from typing import List, Set, Tuple, Optional, Literal
    from rapidfuzz import fuzz
    from rapidfuzz.distance import ScoreAlignment

    fuzzy_match_index_tuple_set_original: Optional[Set[Tuple[int, int]]] = None
    fuzzy_match_index_tuple_set_translation: Optional[
        Set[Tuple[int, int]]] = None

    def fuzzy_match(
        text_normalized: str, search_list: List[str]
    ) -> Optional[Set[Tuple[int, int]]]:
        fuzzy_match_index_tuple_set: Optional[Set[Tuple[int, int]]] = set()
        for search in search_list:
            queue: List[Tuple[str, int]] = [(text_normalized, 0)]
            while len(queue) > 0:
                text, start_index = queue.pop(0)
                if fuzz.ratio(text, search) > 90:
                    fuzzy_match_index_tuple_set.add(
                        (start_index, start_index + len(text)))
                    break
                elif len(text) >= len(search):
                    match_result: Optional[
                        ScoreAlignment] = fuzz.partial_ratio_alignment(
                        text, search, score_cutoff=90
                    )
                    if match_result is not None:
                        fuzzy_match_index_tuple_set.add(
                            (start_index + match_result.src_start,
                             start_index + match_result.src_end)
                        )
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
                                    text[match_result.src_end:],
                                    start_index + match_result.src_end,
                                )
                            )
        if len(fuzzy_match_index_tuple_set) > 0:
            return fuzzy_match_index_tuple_set
        else:
            return None

    def get_token_index(
        char_index: int,
        char_type: Literal["original", "translation"]
    ) -> Tuple[int, bool]:
        token_start_index_list: List[
            int] = original_token_start_index_list if char_type == "original" else translation_token_start_index_list
        token_end_index_list: List[
            int] = original_token_end_index_list if char_type == "original" else translation_token_end_index_list
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
        fuzzy_match_index_tuple_set: Optional[Set[Tuple[
            int, int]]] = fuzzy_match_index_tuple_set_original if text_type == "original" else fuzzy_match_index_tuple_set_translation
        token_list: List[
            str] = original_token_list if text_type == "original" else translation_token_list
        for fuzzy_match_index_tuple in fuzzy_match_index_tuple_set:
            start, _ = get_token_index(
                fuzzy_match_index_tuple[0],
                text_type
            )
            end, in_token = get_token_index(
                fuzzy_match_index_tuple[1] - 1,
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
        token_list: List[
            str] = original_token_list if text_type == "original" else translation_token_list
        token_start_index_list: List[
            int] = original_token_start_index_list if text_type == "original" else translation_token_start_index_list
        token_end_index_list: List[
            int] = original_token_end_index_list if text_type == "original" else translation_token_end_index_list
        matched_index_tuple_list: List[Tuple[int, int]] = []
        for j, word in enumerate(token_list):
            if word.startswith("<mark>") and word.endswith("</mark>"):
                matched_index_tuple_list.append((j, j + 1))
            elif word.startswith("<mark>"):
                matched_index_tuple_list.append(
                    (j, len(token_list))
                )
            elif word.endswith("</mark>"):
                matched_index_tuple_list[-1] = (matched_index_tuple_list[-1][0],
                                                j + 1)

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

    fuzzy_match_index_tuple_set_original = fuzzy_match(
        original_text_normalized, search_item_list_original
    )
    if len(search_item_list) > 0:
        if fuzzy_match_index_tuple_set_original is not None:
            search_result_index_tuple_set = fuzzy_match(
                original_text_normalized, search_item_list
            )
            if search_result_index_tuple_set is not None:
                fuzzy_match_index_tuple_set_original.update(
                    search_result_index_tuple_set)
        else:
            fuzzy_match_index_tuple_set_translation = fuzzy_match(
                translation_normalized, search_item_list
            )

    if fuzzy_match_index_tuple_set_original is not None:
        bold_matched_text(
            "original",
        )
        return simplify_search_result("original")
    elif fuzzy_match_index_tuple_set_translation is not None:
        bold_matched_text(
            "translation",
        )
        return simplify_search_result("translation")
    else:
        return None
$$ LANGUAGE plpython3u;
"""
)
